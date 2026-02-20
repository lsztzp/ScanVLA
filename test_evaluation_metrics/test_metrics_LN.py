import argparse
import json
from os.path import join

from tqdm import tqdm
import argparse
import os.path as osp
from mmengine.dist import (collect_results, get_dist_info, get_rank, init_dist,
                           master_only)
from mmengine.config import Config
from mmengine.fileio import get_file_backend
from xtuner.registry import BUILDER
from xtuner.model.utils import guess_load_checkpoint
import os
from transformers import AutoTokenizer
import torch
import numpy as np

# 将根目录加入sys.path
import os 
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from projects.ScanVLA_LN.metrics.local_optimal_transport import local_OT


def get_metrics_LBM_single(gt_bbox, predict_bbox, unpredicted_mask, window_size=0):
    # 将 gt_bbox 转换为 tensor（如果是 array），并移到 CPU
    if not isinstance(gt_bbox, torch.Tensor):
        gt_bbox = torch.tensor(gt_bbox, dtype=torch.float32)
    else:
        gt_bbox = gt_bbox.cpu().float()
    
    # 处理 predict_bbox: 如果是 (1, L, 4)，去掉 batch 维度变为 (L, 4)，并移到 CPU
    if predict_bbox.dim() == 3:
        predict_bbox = predict_bbox.squeeze(0)  # (1, L, 4) -> (L, 4)
    predict_bbox = predict_bbox.cpu().float()
    
    # 处理 unpredicted_mask: 如果是列表，转换为 tensor；如果是 (1, N, 4)，需要处理
    if isinstance(unpredicted_mask, list):
        unpredicted_mask = torch.tensor(unpredicted_mask, dtype=torch.float32)
    elif isinstance(unpredicted_mask, torch.Tensor):
        unpredicted_mask = unpredicted_mask.cpu().float()
        # 如果是 (1, N, 4)，需要 squeeze 或 reshape
        if unpredicted_mask.dim() == 3:
            # 如果是 (1, N, 4)，取第一个维度并保持 (N, 4) 或压缩为 (N,)
            if unpredicted_mask.shape[2] == 4:
                # 如果最后一维是4，可能需要取平均或只取第一个通道
                unpredicted_mask = unpredicted_mask.squeeze(0)[:, 0]  # (1, N, 4) -> (N, 4) -> (N,)
            else:
                unpredicted_mask = unpredicted_mask.squeeze(0).squeeze(-1)  # (1, N, 1) -> (N,)
        elif unpredicted_mask.dim() == 2:
            # 如果是 (N, 4)，取第一个通道或平均
            unpredicted_mask = unpredicted_mask[:, 0]  # (N, 4) -> (N,)
    
    # 确保 unpredicted_mask 的形状为 (L,) 以便广播
    unpredicted_mask = unpredicted_mask.squeeze()
    if unpredicted_mask.dim() == 0:
        unpredicted_mask = unpredicted_mask.unsqueeze(0)
    
    # 确保长度匹配
    L = min(gt_bbox.shape[0], predict_bbox.shape[0], unpredicted_mask.shape[0])
    gt_bbox = gt_bbox[:L, :4]
    predict_bbox = predict_bbox[:L, :4]
    unpredicted_mask = unpredicted_mask[:L]
    
    # 计算 predicted_mask
    predicted_mask = 1 - unpredicted_mask
    # 扩展 predicted_mask 以匹配 bbox 的维度 (L,) -> (L, 1) 以便广播
    predicted_mask = predicted_mask.unsqueeze(-1)  # (L,) -> (L, 1)

    # 是否需要分块（local OT）
    # use_local_OT = False
    use_local_OT = True

    if use_local_OT:
        # tmp_trace1 对应 gt_bbox，tmp_trace2 对应 predict_bbox
        # 这里不需要 batch 维度：都用 (L, 4)
        valid = (predicted_mask.squeeze(-1) > 0.5)  # (L,)
        tmp_trace1 = gt_bbox[valid]          # (L', 4)
        tmp_trace2 = predict_bbox[valid]     # (L', 4)

        if tmp_trace1.numel() == 0 or tmp_trace2.numel() == 0:
            return 0.0

        # local_OT 里假设 p = D.shape[1], m = D.shape[2]（通常 p <= m），这里按长度自动交换
        if tmp_trace1.shape[0] <= tmp_trace2.shape[0]:
            trace1 = tmp_trace1
            trace2 = tmp_trace2
        else:
            trace1 = tmp_trace2
            trace2 = tmp_trace1

        seg_loss_list = []
        nseg = int(np.ceil(trace1.shape[0] / 20.0))
        for seg_idx in range(nseg):
            tmp_const = 20.0 * trace2.shape[0] / max(trace1.shape[0], 1)
            seg1 = trace1[seg_idx * 20:(seg_idx + 1) * 20, :4]  # (p, 4)

            left = int(np.floor(seg_idx * tmp_const))
            right = int(np.ceil((seg_idx + 1) * tmp_const))
            seg2 = trace2[left:right, :4]  # (m, 4)

            if seg1.shape[0] == 0 or seg2.shape[0] == 0:
                continue

            # D: (p, m) -> local_OT expects (B, p, m)
            D = torch.abs(seg1[:, None, :] - seg2[None, :, :]).mean(dim=-1)  # (p, m)
            T = local_OT(D.unsqueeze(0), window=window_size)[0]              # (p, m)
            seg_cost = (T * D).sum() / seg1.shape[0]
            if not torch.isnan(seg_cost):
                seg_loss_list.append(seg_cost.item())

        lbm = float(np.mean(np.array(seg_loss_list))) if len(seg_loss_list) else 0.0
    else:
        lbm = ((torch.abs(gt_bbox - predict_bbox) * predicted_mask).sum() / (predicted_mask.sum() * 4)).item()
    
    return lbm

def get_metrics_LBM(predictions, window_size=0 ):
    results = []
    for predict in tqdm(predictions):
        predict_bbox = predict['predict_bbox']
        gt_bbox = predict['gt_bbox']
        unpredicted_mask = predict['unpredicted_mask']

        # parameter, k=0 or k=1
        cur_lbm = get_metrics_LBM_single(gt_bbox, predict_bbox, unpredicted_mask, window_size=window_size)

        results.append(cur_lbm)
    LBM_score = np.mean(results)
    return {'LBM': LBM_score}

def parse_args():
    parser = argparse.ArgumentParser(description='toHF script')
    parser.add_argument('--config', default='projects/ScanVLA_LN/configs/ScanVLA_LN.py', help='config file name or path.')
    
    parser.add_argument('--pthmodel',default='pretrained/checkpoints/LN_LBM_0142.pth', help='pth model file')
    
    parser.add_argument('--dataset_dir', type=str, default='/data/lyt/01-Datasets/01-ScanPath-Datasets/LN/', help='save folder name')

    parser.add_argument('--test_file', type=str, default='coco_val/coco_val.json', help='save folder name')

    parser.add_argument('--window_size', type=int, default=1, help='save folder name')
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    cfg = Config.fromfile(args.config)
    dataset = BUILDER.build(cfg.LN_gaze_dataset)
    model = BUILDER.build(cfg.model)
    
    pretrained_state_dict = guess_load_checkpoint(args.pthmodel)
    model.load_state_dict(pretrained_state_dict, strict=False)
    print(f'Load pretrained weight from {args.pthmodel}')
    model._merge_lora()
    del pretrained_state_dict
    model.mllm.transfer_to_hf = True
    model = model.eval().cuda()

    tokenizer_path="/data/lyt/03-Repositories/02-others/03-MultiModality/Qwen3-VL-2B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # 数据预处理
    test_refgazes = json.load(open(join(args.dataset_dir, args.test_file), mode='r'))
    test_refgazes = list(test_refgazes.values())  # 获取字典所有值的列表

    # 轨迹生成
    res = []
    cnt = 0
    for fix in tqdm(test_refgazes):
        cnt += 1
        if cnt > 500: #先测试50条测试集的成绩   
            break

        input_dict = {}

        Image_path = fix['image_path']

        image = dataset._read_image(Image_path) #1
        assert image is not None
        image_data = dataset._process_single_image(image, dataset.single_image_mode)
        input_dict.update(image_data)
        image_token_str = dataset._create_image_token_string(image_data['num_image_tokens'])
        vp_token_str = ''

        fix_caption_information = torch.load(fix['record_path'])
        fix.update(fix_caption_information)
        
        caption_with_token = '<|object_ref_start|> ' + fix['caption']+ '<|object_ref_end|>'

        # 添加模板
        text = "<image>\n" + "Please generate an eye movement scanpath to achieve the following image caption: {class_name}.".format(class_name=caption_with_token)
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
            }

        with torch.no_grad():
            generate_output = model.mllm(mm_inputs, None, 'loss')

            input_ids = mm_inputs['input_ids']
            hidden_states = generate_output.hidden_states
            hidden_states = hidden_states[-1]
            
            ref_encode_see = model.tokenizer.decode(input_ids[0])
            # 找到视觉特征的嵌入位置
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
                hidden_states[b, ref_start_token_indices[b]+1:ref_end_token_indices[b], :] for b in range(input_ids.size(0))])
            pred_embeddings_vl = pred_embeddings_vl[0] #需要保证bs=1

            outs = model.Scanpath_Decoder(vl_guidance_feats=pred_embeddings_vl, pred_embeddings_vision=pred_embeddings_vision)
        
            point_outputs = model.point_predictor(outs).permute(2,1,0) #(B,L,6)
            point_outputs = model.point_activation(point_outputs)

            start, end = ref_start_token_indices[0], ref_end_token_indices[0] #这里需要确保bs==1

            predict_bbox = point_outputs[:, :, :4]
            predict_center_x = point_outputs[:, :, 4]
            predict_center_y = point_outputs[:, :, 5]

            tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
            cleaned_tokens = [token.replace('Ġ', '') for token in tokens]  #取代tokens里的G
            cleaned_tokens = cleaned_tokens[start+1:end]
            unpredicted_mask, token_utterance_indices = dataset.get_unpredicted_mask_with_cleaned_tokens(cleaned_tokens, fix['utterance'])

            # 根据位置将注视点信息进行转换
            fixation_x_tokens, fixation_y_tokens, bbox = dataset.transform_fixation_by_pos(fix['fixation_x'], fix['fixation_y'], fix['bbox_np'], token_utterance_indices)
            # 根据奇异值的位置对fixation,bbox以及unpredicted_mask进行更新
            gt_center_x, gt_center_y, gt_bbox, unpredicted_mask = dataset.update_fixation_and_bbox_by_singular_values(fixation_x_tokens, fixation_y_tokens, bbox, unpredicted_mask)

            caption = fix['caption']
            res.append({'IMG_ID': fix['image_id'], 
                    'predict_bbox': predict_bbox, 
                    'predict_center_x': predict_center_x, 
                    'predict_center_y': predict_center_y, 
                    'gt_bbox': gt_bbox, 
                    'gt_center_x': gt_center_x, 
                    'gt_center_y': gt_center_y,
                    'unpredicted_mask': unpredicted_mask,
                    'caption': caption
                })

    score = get_metrics_LBM(res, window_size = args.window_size)
    print(score)
    print('done')
