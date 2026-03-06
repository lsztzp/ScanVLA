import argparse
import json
from os.path import join

from tqdm import tqdm
import argparse
import os.path as osp
from mmengine.dist import (collect_results, get_dist_info, get_rank, init_dist,
                           master_only)
from xtuner.registry import BUILDER
from xtuner.model.utils import guess_load_checkpoint
from mmengine.config import Config
from transformers import AutoTokenizer
import torch
import numpy as np
import time
from thop import profile

# 将根目录加入sys.path
import os 
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from projects.ScanVLA_RefCOCOGaze_Ablation.metrics.eval_metrics import get_metrics
from projects.ScanVLA_RefCOCOGaze_Ablation.datasets.ScanVLA_RefCOCOGaze import pos_to_fixation

def parse_args():
    parser = argparse.ArgumentParser(description='toHF script')   
    parser.add_argument('--config', default='projects/ScanVLA_RefCOCOGaze_Ablation/configs/ScanVLA_RefCOCOGaze_Baseline.py', help='config file name or path.')    

    parser.add_argument('--pthmodel',default='work_dirs/ScanVLA_RefCOCOGaze_wo_txt_loss/iter_51072.pth', help='pth model file')

    parser.add_argument('--img_dir', type=str, default='/data/lyt/01-Datasets/01-ScanPath-Datasets/ART_data/data/images_512X320/', help='save folder name')
    
    parser.add_argument('--dataset_dir', type=str, default='/data/lyt/01-Datasets/01-ScanPath-Datasets/ART_data/data/', help='save folder name')
    parser.add_argument('--test_file', type=str, default='refcocogaze_test_correct_512X320.json', help='save folder name')
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    cfg = Config.fromfile(args.config)
    dataset = BUILDER.build(cfg.refcoco_gaze_dataset)
    model = BUILDER.build(cfg.model)
    
    pretrained_state_dict = guess_load_checkpoint(args.pthmodel)
    model.load_state_dict(pretrained_state_dict, strict=False)
    print(f'Load pretrained weight from {args.pthmodel}')
    model._merge_lora()
    del pretrained_state_dict
    model.mllm.transfer_to_hf = True
    model = model.eval().cuda()

    # total, trainable, non_trainable = count_model_params(model)

    # tokenizer_path="/data/lyt/03-Repositories/02-others/03-MultiModality/Qwen3-VL-2B-Instruct"
    # tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer = model.tokenizer
    
    # 数据预处理
    test_refgazes = json.load(open(join(args.dataset_dir, args.test_file), mode='r'))
    test_refs = []
    test_ref_set = set()
    for case in test_refgazes:
        if case['REF_ID'] not in test_ref_set:
            test_ref_set.add(case['REF_ID'])
            test_refs.append({'REF_ID': case['REF_ID'], 'IMAGEFILE': case['IMAGEFILE'], 'REF_WORDS': case['REF_WORDS'], 'REF_SENTENCE': case['REF_SENTENCE']})

    # 轨迹生成
    res = []
    # start_time = time.time()
    for ref in tqdm(test_refs):
        input_dict = {}

        Image_path = join(args.img_dir, ref['IMAGEFILE'])

        image = dataset._read_image(Image_path) #1

        assert image is not None
        image_data = dataset._process_single_image(image, dataset.single_image_mode)
        input_dict.update(image_data)

        image_token_str = dataset._create_image_token_string(image_data['num_image_tokens'])

        vp_token_str = ''

        # 获得所需要的指代表达中的掩码
        ref['REF_SENTENCE'] = '<|object_ref_start|> ' + ref['REF_SENTENCE'] + '<|object_ref_end|>'
        ref_encode_see = tokenizer.tokenize(ref['REF_SENTENCE'], add_special_tokens=False)
        ref_encode = tokenizer(ref['REF_SENTENCE'], add_special_tokens=False, return_offsets_mapping=True)
        ref_offset = ref_encode['offset_mapping']
        ref_offset_mask = pos_to_fixation(ref_offset, ref['REF_SENTENCE'])
        ref_offset_mask = torch.tensor(ref_offset_mask).cuda()

        # 添加模板
        text = "<image>\n" + "Please generate an eye movement scanpath to achieve the following objectives: {class_name}.".format(class_name=ref['REF_SENTENCE'])

        text = text.replace('<image>', image_token_str + vp_token_str)
        input_text = ''
        template = '<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n'
        input_text += template.format(input=text)
        ids = tokenizer.encode(input_text)
        ids = torch.tensor(ids).cuda().unsqueeze(0)

        attention_mask = torch.ones_like(ids, dtype=torch.bool)

        ids = ids.repeat(1,1)
        attention_mask = attention_mask.repeat(1,1)

        bs, seq_len = ids.shape
        position_ids = torch.arange(seq_len).unsqueeze(0).long().repeat(bs, 1)

        # 输入内容
        mm_inputs = {
                'pixel_values': [input_dict['pixel_values'].cuda()],
                'g_pixel_values': [input_dict['g_pixel_values'].cuda()],
                'image_grid_thw': [input_dict['image_grid_thw'].cuda()],
                'input_ids': ids.cuda(),
                'attention_mask': attention_mask.cuda(),
                'position_ids': position_ids.cuda(),
                'past_key_values': None,
                'labels': None,
                'prompt_masks': None,
                # 'vp_overall_mask': input_dict['vp_overall_mask'],
            }

        with torch.no_grad():
            # flops, params = profile(model.mllm, inputs=(mm_inputs, None, 'loss'))

            # # 转换成 TOPS 相关单位
            # GFLOPs = flops / 1e9
            # TOPs   = flops / 1e12   # 1 T = 10¹²

            # print(f"GFLOPs: {GFLOPs:.2f}")
            # print(f"TOPs:   {TOPs:.4f}")
            generate_output = model.mllm(mm_inputs, None, 'loss')
        # break

        input_ids = mm_inputs['input_ids']
        hidden_states = generate_output.hidden_states
        hidden_states = hidden_states[-1]  #1

        # 找到视觉特征的开始和结束
        vision_start_token_mask = input_ids == model.vision_start_token_idx
        vision_end_token_mask = input_ids == model.vision_end_token_idx
        vision_start_token_indices = torch.argmax(vision_start_token_mask.int(), dim=1)
        vision_end_token_indices = torch.argmax(vision_end_token_mask.int(), dim=1)
        pred_embeddings_vision = torch.stack([
            hidden_states[b, vision_start_token_indices[b]+1:vision_end_token_indices[b], :] for b in range(input_ids.size(0))]).permute(1,0,2)
        
        # 找到ref的开始和结束位置
        ref_start_token_mask = input_ids == model.start_token_idx
        ref_end_token_mask = input_ids == model.end_token_idx
        ref_start_token_indices = torch.argmax(ref_start_token_mask.int(), dim=1)
        ref_end_token_indices = torch.argmax(ref_end_token_mask.int(), dim=1)
        pred_embeddings_vl = torch.stack([
            hidden_states[b, ref_start_token_indices[b]:ref_end_token_indices[b]+1, :] for b in range(input_ids.size(0))])
        pred_embeddings_vl = pred_embeddings_vl[0] #需要保证bs=1

        outs = model.Scanpath_Decoder(vl_guidance_feats=pred_embeddings_vl, pred_embeddings_vision=pred_embeddings_vision)

        # 预测y,x,token
        y_mu, x_mu = model.generator_y_mu(outs).permute(2,1,0), model.generator_x_mu(outs).permute(2,1,0)
        token_predict = model.token_predictor(outs) 
        token_predict = token_predict.unsqueeze(dim=0).permute(0,2,1,3) #前面的1代表bs=1
        token_states = torch.argmax(token_predict, dim=-1)

        x_mu = model.activation(x_mu)  #(B,L,4)
        y_mu = model.activation(y_mu)  #(B,L,4)
        x_mu = x_mu * 512
        y_mu = y_mu * 320
        
        ref_offset_mask = ref_offset_mask.unsqueeze(0)
        scanpaths_x = x_mu * ref_offset_mask.unsqueeze(-1).expand_as(x_mu)
        scanpaths_y = y_mu * ref_offset_mask.unsqueeze(-1).expand_as(y_mu)
        
        a,b = [],[]
        for i in range(scanpaths_x.shape[0]):
            word_valid_length = ref_offset_mask[i].sum()
            scanpaths_x_each = scanpaths_x[i]
            scanpaths_y_each = scanpaths_y[i]
            token_states_each = token_states[i]
            for j in range(word_valid_length):
                tmp_x,tmp_y = [],[]
                for k in range(len(token_states_each[j])):
                    if token_states_each[j][k] == 0:
                        tmp_x.append(scanpaths_x_each[j][k].item())
                        tmp_y.append(scanpaths_y_each[j][k].item())
                    else:
                        break
                a.append(tmp_x)
                b.append(tmp_y)
        res.append({'REF_ID': ref['REF_ID'], 'X': a, 'Y': b , 'TERMINATIONS': -1, 'REPEAT_ID': 0})

    # end_time = time.time()
    # elapsed_time = end_time - start_time  # 单位：秒
    
    # 6. 格式化输出
    # print(f"单次推理耗时：{elapsed_time:.6f} 秒")
    # print(f"单次推理耗时：{elapsed_time*1000:.2f} 毫秒")

    score = get_metrics(res)
    print(score)
    print('done')
