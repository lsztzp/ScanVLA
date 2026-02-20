import argparse
import json
from os.path import join

from tqdm import tqdm
import argparse

from xtuner.registry import BUILDER
from xtuner.model.utils import guess_load_checkpoint
from mmengine.config import Config

from transformers import AutoTokenizer
import torch

import numpy as np
import pickle

import os 
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# 这些函数是通用的，因此只在ScanVLA_COCOSearch18下保存了一份，这里在这里导入
from projects.ScanVLA_COCOSearch18.metrics.metrics import postprocessScanpaths, get_seq_score, get_ed, get_semantic_ed, get_semantic_seq_score,compute_mm, compute_spatial_metrics_by_step
from projects.ScanVLA_COCOSearch18.metrics.metrics import get_seq_score_time,get_ed_time,get_semantic_seq_score_time,get_semantic_ed_time

def run_model(model, tokenizer, task_name, Image_path, num_samples=1):
    input_dict = {}

    image = dataset._read_image(Image_path) #1
    image = image.resize((520, 320)) #固定到指定大小

    assert image is not None
    image_data = dataset._process_single_image(image, dataset.single_image_mode)
    input_dict.update(image_data)
    image_token_str = dataset._create_image_token_string(image_data['num_image_tokens'])
    vp_token_str = ''

    # 保持VLM冻结，使用更合适的prompt.
    # category_with_token = '<|object_ref_start|> ' + task_name+ '<|object_ref_end|>'
    category_with_token = task_name

    # 添加模板
    text = "<image>\n" + "Please segment {class_name} in this image.".format(class_name=category_with_token)

    text = text.replace('<image>', image_token_str + vp_token_str)
    input_text = ''
    template = '<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\nSure, it is [SEG].<|im_end|>\n'

    input_text += template.format(input=text)
    ids = tokenizer.encode(input_text)
    ids = torch.tensor(ids).cuda().unsqueeze(0)

    # words_seen = tokenizer.decode(ids[0]) #1

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

    hidden_states = generate_output.hidden_states[-1]

    seg_token_mask = input_ids == model.seg_token_idx
    end_token_indices = torch.argmax(seg_token_mask.int(), dim=1) 
    pred_embeddings_vl = hidden_states[:, end_token_indices, :][0]
    pred_embeddings_vl = pred_embeddings_vl

    vision_start_token_mask = input_ids == model.vision_start_token_idx
    vision_end_token_mask = input_ids == model.vision_end_token_idx
    vision_start_token_indices = torch.argmax(vision_start_token_mask.int(), dim=1)
    vision_end_token_indices = torch.argmax(vision_end_token_mask.int(), dim=1)
    pred_embeddings_vision = torch.stack([
        hidden_states[b, vision_start_token_indices[b]+1:vision_end_token_indices[b], :] for b in range(input_ids.size(0))]).permute(1,0,2)

    outs = model.Scanpath_Decoder(vl_guidance_feats=pred_embeddings_vl, 
                                  pred_embeddings_vision=pred_embeddings_vision)

    y_mu = model.activation(model.generator_y_mu(outs)).flatten().cpu()
    x_mu = model.activation(model.generator_x_mu(outs)).flatten().cpu() 
    t_mu = model.activation_time(model.generator_t_mu(outs)).flatten().cpu() 
    token_predict = model.softmax(model.token_predictor(outs).view(outs.shape[0],-1,2)).cpu()
    
    x_mu = x_mu * 512
    y_mu = y_mu * 320

    token_states = torch.argmax(token_predict, dim=-1)

    scanpaths = []
    for i in range(num_samples):
        ys_i = y_mu.squeeze()
        xs_i = x_mu.squeeze()
        ts_i = t_mu.squeeze()
        token_type = token_states.squeeze()

        ys_i = ys_i
        xs_i = xs_i
        ts_i = ts_i
        token_type = torch.cat([torch.tensor([0], dtype=token_type.dtype, device=token_type.device), token_type[1:]])

        scanpath = []
        for tok, y, x, t in zip(token_type, ys_i, xs_i, ts_i):
            if tok == 0:
                scanpath.append([min(320-1, y.item()),min(512 - 1, x.item()), t.item()])
            else:
                break
        scanpaths.append(np.array(scanpath))
    return scanpaths


def parse_args():
    parser = argparse.ArgumentParser(description='toHF script')
    parser.add_argument('--config', default='projects/ScanVLA_COCOSearch18_ZeroGaze/configs/ScanVLA_ZeroGaze_laptop.py', help='config file name or path.')

    parser.add_argument('--max_len', type=int, default=7, help='max_len')

    parser.add_argument('--image_dir_present', type=str, default='/data/lyt/01-Datasets/01-ScanPath-Datasets/coco_search18/raw/COCOSearch18/images/', help='save folder name')
    parser.add_argument('--image_dir_absent', type=str, default='/data/lyt/01-Datasets/01-ScanPath-Datasets/coco_search18/raw/COCOSearch18/coco_search18_images_TA/', help='save folder name')

    parser.add_argument('--dataset_dir', type=str, default='/data/lyt/01-Datasets/01-ScanPath-Datasets/coco_search18/raw/COCOSearch18/dataset_json/', help='save folder name')
    
    # need to modify according to the different task
    parser.add_argument('--zerogaze', default=True, action='store_true', help="ZeroGaze setting flag")
    parser.add_argument('--task', default='laptop', type=str, help="if evaluation is in ZeroGaze setting, the unseen target to evaluate the model")
    parser.add_argument('--pthmodel',default='/data/lyt/03-Repositories/01-ours/ScanVLA/ScanHisVLA/work_dirs/ScanHisVLA_TP_ZeroGaze_laptop/iter_145152.pth', help='pth model file')

    parser.add_argument('--num_samples', default=1, type=int, help="Number of scanpaths sampled per test case")

    parser.add_argument('--condition', type=str, default='present', help='condition')
 
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    cfg = Config.fromfile(args.config)
    dataset = BUILDER.build(cfg.COCOSearch18_gaze_dataset)
    model = BUILDER.build(cfg.model)
    # Load pretrained weights
    pretrained_state_dict = guess_load_checkpoint(args.pthmodel)
    model.load_state_dict(pretrained_state_dict, strict=False)

    print(f'Load pretrained weight from {args.pthmodel}')
    model._merge_lora()
    del pretrained_state_dict
    model.mllm.transfer_to_hf = True
    model = model.eval().cuda()

    # tokenizer_path="/data/lyt/03-Repositories/02-others/03-MultiModality/Qwen3-VL-2B-Instruct"
    # tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer = model.tokenizer

    # Prepare test data
    dataset_root = args.dataset_dir
    img_ftrs_dir = args.image_dir_present
    if args.condition == 'absent':
        img_ftrs_dir = args.image_dir_absent
    max_len = args.max_len
    fixation_path = join(dataset_root, 'coco_search18_fixations_TP_test.json')
    if args.condition == 'absent':
        fixation_path = join(dataset_root, 'coco_search18_fixations_TA_test.json')
    with open(fixation_path) as json_file:
        human_scanpaths = json.load(json_file)
    test_target_trajs = list(filter(lambda x: x['split'] == 'test' and x['condition']==args.condition, human_scanpaths))

    if args.zerogaze:
        assert args.task in ["bottle", "bottle", "bowl", "car", "chair", "clock", "cup", "fork", "keyboard", "knife", "laptop", "microwave",
                            "mouse", "oven", "potted plant", "sink", "stop sign", "toilet", "tv"], "current task is {} must be in {}".format(args.task, ["bottle", "bottle", "bowl", "car", "chair", "clock", "cup", "fork", "keyboard", "knife", "laptop", "microwave",
                            "mouse", "oven", "potted plant", "sink", "stop sign", "toilet", "tv"])
        test_target_trajs = list(filter(lambda x: x['task'] == args.task.replace('_', ' '), test_target_trajs))
        print("Zero Gaze on", args.task.replace('_', ' '))

    t_dict = {}
    for traj in test_target_trajs:
        key = 'test-{}-{}-{}-{}'.format(traj['condition'], traj['task'],
                                     traj['name'][:-4], traj['subject'])

        t_dict[key] = np.array(traj['T'])

    test_task_img_pairs = np.unique([traj['task'] + '_' + traj['name'] + '_' + traj['condition'] for traj in test_target_trajs])

    pred_list = []
    print('Generating {} scanpaths per test case...'.format(args.num_samples))

    for target_traj in tqdm(test_task_img_pairs):
        task_name, name, condition = target_traj.split('_')
        Image_path = join(img_ftrs_dir, task_name.replace(' ', '_'), name)

        scanpaths = run_model(model, tokenizer, task_name, Image_path, num_samples=args.num_samples)
            
        for idx, scanpath in enumerate(scanpaths):
            pred_list.append((task_name, name, condition, idx+1, scanpath))

    predictions = postprocessScanpaths(pred_list)
    fix_clusters = torch.load('./projects/ScanVLA_COCOSearch18/metrics/clusters_cp.pt')

    print("Calculating Sequence Score...")
    seq_score = get_seq_score(predictions, fix_clusters, max_len)
    FED = get_ed(predictions, fix_clusters, max_len)
    print('Sequence Score : {:.3f}, FED : {:.3f}'.format(seq_score, FED))

    print("Calculating SemSS,SemFED")
    if args.condition == 'present':
        with open('./projects/ScanVLA_COCOSearch18/metrics/segmentation_map_dir/SemSS/test_TP_Sem.pkl', "rb") as r:
            fixations_dict = pickle.load(r)
            r.close()
    elif args.condition == 'absent':
        with open('./projects/ScanVLA_COCOSearch18/metrics/segmentation_map_dir/SemSS/test_TA_Sem.pkl', "rb") as r:
            fixations_dict = pickle.load(r)
            r.close()
    SemSS = get_semantic_seq_score(predictions, fixations_dict, max_len, './projects/ScanVLA_COCOSearch18/metrics/segmentation_map_dir/SemSS/stuffthing_maps')
    SemFED = get_semantic_ed(predictions, fixations_dict, max_len, './projects/ScanVLA_COCOSearch18/metrics/segmentation_map_dir/SemSS/stuffthing_maps')
    print('SemSS : {:.3f}, SemFED : {:.3f}'.format(SemSS, SemFED))

    print("Calculating MM")
    if args.condition == 'absent':
        for x in test_target_trajs:
            x['X'] = [a / 1680 * 512 for a in x['X']]
            x['Y'] = [a / 1050 * 320 for a in x['Y']]
    mm = compute_mm(test_target_trajs, predictions, 512, 320)
    mm = mm[:-1].mean()
    print('MM : {:.3f}'.format(mm))

    print("Calculating CC,NSS")
    cc, nss = compute_spatial_metrics_by_step(predictions, test_target_trajs)
    print('CC : {:.3f}, NSS : {:.3f}'.format(cc, nss))

    # Time-aware metrics
    print("Calculating Time-aware metrics...")
    seq_score_t = get_seq_score_time(predictions, fix_clusters, max_len, t_dict)
    fed_t = get_ed_time(predictions,fix_clusters, max_len, t_dict)
    SemSS_t = get_semantic_seq_score_time(predictions,fixations_dict,max_len,'./projects/ScanVLA_COCOSearch18/metrics/segmentation_map_dir/SemSS/stuffthing_maps')
    SenFED_t = get_semantic_ed_time(predictions, fixations_dict, max_len, './projects/ScanVLA_COCOSearch18/metrics/segmentation_map_dir/SemSS/stuffthing_maps')

    print('Sequence Score_T : {:.3f}, FED_T : {:.3f}'.format(seq_score_t, fed_t))
    print('SemSS_T : {:.3f}, SemFED_T : {:.3f}'.format(SemSS_t, SenFED_t))

    print('Done')
