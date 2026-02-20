import argparse
import json
from os.path import join

from tqdm import tqdm
import argparse

from xtuner.registry import BUILDER
from xtuner.model.utils import guess_load_checkpoint
from mmengine.config import Config

import torch
import random 
import numpy as np
import pickle

import scipy.io as scio
import multimatch_gaze as m

import os 
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from projects.ScanVLA_AiR.metrics.python.H_MM_Distance_compute import H_MM_Distance
from projects.ScanVLA_AiR.metrics.python.ScanMatch import ScanMatch

from projects.ScanVLA_AiR.metrics.python.SS import SS_Score

ScanMatchInfo_osie = scio.loadmat('projects/ScanVLA_AiR/metrics/python/OSIE_ScanMatchInfo.mat')['ScanMatchInfo']
gtspath = '/data/lyt/01-Datasets/01-ScanPath-Datasets/AiR/air_processed'

def score_seq(pre, gt, metrics=('scanmatch', 'tde', 'mutimatch')):
    image_size = [600, 800]
    ScanMatchInfo = ScanMatchInfo_osie

    scores = {}
    if 'scanmatch' in metrics:
        scores['scanmatch'] = ScanMatch(pre.astype(int), gt.astype(int), ScanMatchInfo)
    if 'tde' in metrics:
        tde_h, tde_m = H_MM_Distance(pre.astype(int), gt.astype(int))
        scores['tde_h'], scores['tde_m'] = np.array(tde_h), np.array(tde_m)
    if 'mutimatch' in metrics:
        pre = np.array([(loc[0], loc[1], random.random()) for loc in pre],
                       [('start_x', float), ('start_y', float), ('duration', float)]).view(np.recarray)

        gt = np.array([(loc[0], loc[1], random.random()) for loc in gt],
                      [('start_x', float), ('start_y', float), ('duration', float)]).view(np.recarray)
        

        scores['mutimatch'] = np.array(m.docomparison(pre, gt, screensize=[image_size[0], image_size[1]]))[:-1]
    return scores

def score_all_gts(pred_fixation, gt_fixations, metrics=('scanmatch', 'tde', 'mutimatch')):

    scores_all_gts = {}
    count = 0

    for n in range(len(gt_fixations)):
        gt_fixation = gt_fixations[n].astype(float)
        if len(gt_fixation) >= 3 and len(pred_fixation) >= 3:

            scores = score_seq(pred_fixation.astype(int), gt_fixation.astype(int), metrics)

            for metric, score in scores.items():
                if metric in scores_all_gts:
                    scores_all_gts[metric] += score
                else:
                    scores_all_gts[metric] = score
            count += 1

    if count != 0:
        for metric, score in scores_all_gts.items():
            scores_all_gts[metric] /= count

    return scores_all_gts


def get_score_filename(pred_fixation, file_name, performance, metrics=('scanmatch', 'tde', 'mutimatch')):
    if performance:
        performance = "right"
    else:
        performance = "wrong"

    gt_path = os.path.join(gtspath, file_name + '.pkl')
    with open(gt_path, "rb") as f:
        gt_fixations = pickle.load(f)

    # img_size = gt_fixations["img_size"][0]
    img_size = gt_fixations["img_size"]
    gt_fixations = gt_fixations[f'fixations_{performance}']

    if len(gt_fixations) == 0:
        print("huaile")
        return {}
    # matlab 读取 格式问题
    elif len(gt_fixations.shape) == 2:
        gt_fixations = gt_fixations[0]
    print(gt_fixations.shape)
    h, w = img_size[0], img_size[1]

    pred_fixation = np.clip(pred_fixation, 0.0, 0.98)

    pred_fixation[:, 0] *= 600
    pred_fixation[:, 1] *= 800
    for n in range(len(gt_fixations)):
        gt_fixation = gt_fixations[n].astype(float)
        gt_fixation[:, 0] /= h
        gt_fixation[:, 1] /= w
        gt_fixation = np.clip(gt_fixation, 0.0, 0.98)
        gt_fixation[:, 0] *= 600
        gt_fixation[:, 1] *= 800
        gt_fixations[n] = gt_fixation

    ans = score_all_gts(pred_fixation, gt_fixations, metrics)
    # 计算SS指标
    # SS = SS_Score(pred_fixation, gt_fixations)
    # ans['SS'] = SS
    return ans

def get_metrics(pred_list, metrics=('scanmatch', 'tde', 'mutimatch')):
    val_performance, seq_count  = {}, 0
    for _, (question_id, performance, idx, scanpath) in enumerate(pred_list):
        file_name = question_id
        pred_fixation = scanpath 
        scores = get_score_filename(pred_fixation, file_name, performance, metrics=metrics)

        if scores:
            seq_count += 1
            for metric, score in scores.items():
                if metric in val_performance:
                    val_performance[metric] += score
                else:
                    val_performance[metric] = score
    if seq_count > 0:
        for metric, _ in val_performance.items():
            val_performance[metric] /= seq_count
    return val_performance

def run_model(model, tokenizer, question, Image_path, performance=True, num_samples=1):
    input_dict = {}

    image = dataset._read_image(Image_path) #1
    image = image.resize((520, 320)) #固定到指定大小

    assert image is not None
    image_data = dataset._process_single_image(image, dataset.single_image_mode)
    input_dict.update(image_data)
    image_token_str = dataset._create_image_token_string(image_data['num_image_tokens'])
    vp_token_str = ''

    if '.' == question[-1]:
        question = question[:-1]
    if '?' == question[-1]:
        question = question[:-1] 

    question_with_token = question

    if performance:
        text = "<image>\n" + "Given the visual image and the corresponding question, generate a human scanpath (gaze trajectory) that reflects the cognitive process of correctly answering the following VQA question: {class_name}?".format(class_name=question_with_token)
    else:
        text = "<image>\n" + "Given the visual image and the corresponding question, generate a human scanpath (gaze trajectory) that reflects the cognitive process of incorrectly answering the following VQA question: {class_name}?".format(class_name=question_with_token)

    text = text.replace('<image>', image_token_str + vp_token_str)
    input_text = ''
    template = '<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\nSure, it is [SEG].<|im_end|>\n'

    input_text += template.format(input=text)
    ids = tokenizer.encode(input_text)
    ids = torch.tensor(ids).cuda().unsqueeze(0)

    # 可视化看下编码是否正确
    # words_seen = tokenizer.decode(ids[0]) 

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
    

        outs = model.Scanpath_Decoder(performance=performance,
                                vl_guidance_feats=pred_embeddings_vl, 
                                  pred_embeddings_vision=pred_embeddings_vision)

        y_mu = model.activation(model.generator_y_mu(outs)).flatten().cpu()
        x_mu = model.activation(model.generator_x_mu(outs)).flatten().cpu() 
        token_predict = model.softmax(model.token_predictor(outs)).cpu()

        token_states = torch.argmax(token_predict, dim=-1)

    scanpaths = []
    for i in range(num_samples):
        ys_i = y_mu.squeeze()
        xs_i = x_mu.squeeze()
        token_type = token_states.squeeze()

        ys_i = ys_i
        xs_i = xs_i
        token_type = torch.cat([torch.tensor([0], dtype=token_type.dtype, device=token_type.device), token_type[1:]])

        scanpath = []
        for tok, y, x in zip(token_type, ys_i, xs_i):
            if tok == 0:
                scanpath.append([y.item(), x.item()])
            else:
                break
        scanpaths.append(np.array(scanpath))
    return scanpaths

def parse_args():
    parser = argparse.ArgumentParser(description='toHF script')
    parser.add_argument('--config', default='projects/ScanVLA_AiR/configs/ScanVLA_AiR.py', help='config file name or path.')

    parser.add_argument('--max_len', type=int, default=16, help='save folder name')

    parser.add_argument('--dataset_dir', type=str, default='/data/lyt/01-Datasets/01-ScanPath-Datasets/AiR/', help='save folder name')

    # correct or incorrect
    parser.add_argument('--performance', type=bool, default=True, help='save folder name')
    # parser.add_argument('--performance', type=bool, default=False, help='save folder name')

    parser.add_argument('--pthmodel',default='pretrained/checkpoints/AiR_SM_0421.pth', help='pth model file')

    parser.add_argument('--num_samples', default=1, type=int, help="Number of scanpaths sampled per test case")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    cfg = Config.fromfile(args.config)
    dataset = BUILDER.build(cfg.AiR_gaze_dataset)
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

    fixation_path = args.dataset_dir + "fixations/AiR_fixations_test.json"
    # fixation_path = args.dataset_dir + "fixations/AiR_fixations_validation.json"

    with open(fixation_path) as json_file:
        human_scanpaths = json.load(json_file)
    
    qid_to_sub = {}
    qid_to_img = {}
    qid_to_ques = {}
    for index, fixation in enumerate(human_scanpaths):
        qid_to_sub.setdefault(fixation['question_id'], []).append(index)
        qid_to_img[fixation['question_id']] = fixation['image_id']

        qid_to_ques[fixation['question_id']] = fixation['question']
    qids = list(qid_to_sub.keys())
    
    test_target_trajs = human_scanpaths

    # Prepare test data
    img_ftrs_dir = args.dataset_dir + "stimuli"
    max_len = args.max_len

    pred_list = []
    print('Generating {} scanpaths per test case...'.format(args.num_samples))

    for i, target_traj in tqdm(enumerate(qids)):
        question_id = qids[i]
        img_name = qid_to_img[question_id]
        Image_path = join(img_ftrs_dir, img_name)

        question = qid_to_ques[question_id]

        scanpaths = run_model(model, tokenizer, question, Image_path, performance=args.performance, num_samples=args.num_samples)
            
        for idx, scanpath in enumerate(scanpaths):
            pred_list.append((question_id, args.performance, idx+1, scanpath))

    metrics = ('scanmatch', 'tde', 'mutimatch')
    ans = get_metrics(pred_list, metrics)
    print(ans)

    print('Done')
