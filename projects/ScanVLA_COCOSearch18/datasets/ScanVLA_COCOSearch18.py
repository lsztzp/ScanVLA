import copy
import random
import glob
import json
import logging
import os
from typing import Literal

import torch

from mmengine import print_log
from mmengine.config import Config, ConfigDict
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
# import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils

from xtuner.registry import BUILDER
from xtuner.utils import IGNORE_INDEX
from xtuner.dataset.utils import encode_fn
from xtuner.dataset.map_fns import llava_map_fn

from os.path import join
import torchvision.transforms.functional as F

from .common import SEG_QUESTIONS, ANSWER_LIST

# from projects.glamm.utils import DEFAULT_IMAGE_TOKEN
DEFAULT_IMAGE_TOKEN = "<image>"

from .base import Sa2VABaseDataset
import string


def pos_to_fixation(offset_mapping,text_description):
    # 由于分词的原因，一些单词会被分割为多个token,我们希望每个单词的最后一个token预测注视点，因此需要将每个单词的最后一个token 位置找出来
    original_words = text_description.split(' ')
    word_to_token_mapping = []
    special = 2   #bos和eos
    text_description =text_description+' ' #通过后面的符号进行判断

    for i, (token_start, token_end) in enumerate(offset_mapping):
        if token_start == 0 and token_end == 0:
            if special>0:
                word_to_token_mapping.append(1)
                special -= 1
                continue
            else:
                word_to_token_mapping.append(0)
                continue

        if text_description[token_end]==' ':
            word_to_token_mapping.append(1)
        else:
            word_to_token_mapping.append(0)
    
    word_to_token_mapping[-2] = 1
    return word_to_token_mapping


def pre_process_LN(refgazes, max_lm_length=32, max_pack_length=4):
    for refgaze in refgazes:
        refgaze['REF_SENTENCE'] = '<|object_ref_start|> ' + refgaze['REF_SENTENCE'].strip() + '<|object_ref_end|>'

        pack_x, pack_y = refgaze['PACK_X'], refgaze['PACK_Y']

        context_x,context_y,context_pack,context_order = refgaze['CONTEXT_X'],refgaze['CONTEXT_Y'],refgaze['CONTEXT_PACK'],refgaze['CONTEXT_ORDER']
        pack_len = len(pack_x)
        if pack_len:
            context_x = context_x + pack_x
            context_y = context_y + pack_y
            # context_pack = context_pack + [len(refgaze['CAPTION'].split(' '))-1] * pack_len
            context_pack = context_pack + [len(refgaze['REF_SENTENCE'].split(' '))] * pack_len
            context_order = context_order + list(range(pack_len))
        fixation_x,fixation_y=[[0]*max_pack_length for _ in range(max_lm_length)], [[0]*max_pack_length for _ in range(max_lm_length)]

        x_min, y_min, width, height  = refgaze['BBOX']   #检查顺序
        x_max, y_max = x_min + width, y_min + height

        assert x_min >= 0 and y_min>=0 and x_max<512 and y_max<320
        cnt = 0
        for i in range(len(context_x)):
            pack, order = context_pack[i], context_order[i]
            assert pack < max_lm_length
            if pack == max(context_pack):    #对于最后一个token,只能添加在目标的框中间的fixation
                if context_x[i] >= x_min and context_x[i] <= x_max and context_y[i] >= y_min and context_y[i] <= y_max and cnt < max_pack_length:
                    fixation_x[pack][cnt] = context_x[i]
                    fixation_y[pack][cnt] = context_y[i]
                    cnt += 1
            elif order < max_pack_length:
                fixation_x[pack][order] = context_x[i]       #采取了截取，可能会有找不到目标的样本，是否有更合适的方法
                fixation_y[pack][order] = context_y[i]

        refgaze['fixation_x'] = fixation_x
        refgaze['fixation_y'] = fixation_y

    print('Dataset pre_process Finished')
    return refgazes

def get_num_step2target(X, Y, bbox):
    X, Y = np.array(X), np.array(Y)
    on_target_X = np.logical_and(X > bbox[0], X < bbox[0] + bbox[2])
    on_target_Y = np.logical_and(Y > bbox[1], Y < bbox[1] + bbox[3])
    on_target = np.logical_and(on_target_X, on_target_Y)
    if np.sum(on_target) > 0:
        first_on_target_idx = np.argmax(on_target)
        return first_on_target_idx + 1
    else:
        return 100

def cutFixOnTarget(trajs, target_annos):
    processed_trajs = []
    task_names = np.unique([traj['task'] for traj in trajs])
    if 'condition' in trajs[0].keys():
        trajs = list(filter(lambda x: x['condition'] == 'present', trajs))
    if len(trajs) == 0:
        return
    for task in task_names:
        task_trajs = list(filter(lambda x: x['task'] == task, trajs))
        num_steps_task = np.ones(len(task_trajs), dtype=np.uint8)
        for i, traj in enumerate(task_trajs):
            key = traj['task'] + '_' + traj['img_name']
            bbox = target_annos[key]
            traj_len = get_num_step2target(traj['tgt_seq_x'], traj['tgt_seq_y'], bbox)

            num_steps_task[i] = traj_len
            traj['tgt_seq_x'] = traj['tgt_seq_x'][:traj_len]
            traj['tgt_seq_y'] = traj['tgt_seq_y'][:traj_len]
            traj['tgt_seq_t'] = traj['tgt_seq_t'][:traj_len]
            if traj_len!=100:
                processed_trajs.append(traj)
            # processed_trajs.append(traj)
    # print(cnt,len(trajs))
    print('data cuted')
    return processed_trajs

def fixations2seq(fixations, max_len):
    processed_fixs = []
    num=0
    for fix in fixations:
        if len(fix['X'])<=max_len:
            # 确保 tensor 在 CPU 上创建，避免显存泄漏
            processed_fixs.append({'tgt_seq_y': torch.tensor(np.array(fix['Y'])[:max_len], device='cpu'),
                                   'tgt_seq_x': torch.tensor(np.array(fix['X'])[:max_len], device='cpu'),
                                   'tgt_seq_t': torch.tensor(np.array(fix['T'])[:max_len], device='cpu'),
                                   'task': fix['task'], 'img_name': fix['name']})
        else:
            num+=1
            # 确保 tensor 在 CPU 上创建，避免显存泄漏
            processed_fixs.append({'tgt_seq_y': torch.tensor(np.array(fix['Y'])[-max_len:], device='cpu'),
                                   'tgt_seq_x': torch.tensor(np.array(fix['X'])[-max_len:], device='cpu'),
                                   'tgt_seq_t': torch.tensor(np.array(fix['T'])[-max_len:], device='cpu'),
                                   'task': fix['task'], 'img_name': fix['name']})
    print("Has:%d scanpath over length"%num)
    return processed_fixs


class COCOSearch18Dataset(Sa2VABaseDataset):
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
    IMG_START_TOKEN = '<img>'
    IMG_END_TOKEN = '</img>'

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, 
                fixs, 
                img_dir, 
                tokenizer, 
                prompt_template=None, 
                extra_image_processor=None, 
                special_tokens=None, 
                max_length=8192, 
                num_classes_per_sample=5, 
                arch_type: Literal['intern_vl', 'qwen'] = 'intern_vl',
                single_image_mode=False, 
                preprocessor=None,
                max_lm_lens=16,
                max_pack_len=4,
                repeats:int = 1,
                name: str = 'RefCOCO_Gaze',
                condition="present",
                zerogaze="False",
                task="toilet",
                **kwargs):
        
        Sa2VABaseDataset.__init__(self,
            tokenizer=tokenizer,
            prompt_template=prompt_template,
            max_length=max_length,
            special_tokens=special_tokens,
            arch_type=arch_type,
            preprocessor=preprocessor,
            extra_image_processor=extra_image_processor,
            repeats=repeats,
            name=name
        )
        self.max_lm_lens = max_lm_lens  #在处理数据时进行截取，每个语言描述中最大的token个数
        assert condition in ["present","absent"], "condition must be present or absent"
        if condition=="present":
            fixs = "/data/lyt/01-Datasets/01-ScanPath-Datasets/coco_search18/raw/COCOSearch18/dataset_json/coco_search18_fixations_TP_train.json"
        elif condition=="absent":
            fixs = "/data/lyt/01-Datasets/01-ScanPath-Datasets/coco_search18/raw/COCOSearch18/dataset_json/coco_search18_fixations_TA_train.json"

        fixations_train = json.load(open(fixs, mode='r'))
        fixations_train = fixations2seq(fixations=fixations_train, max_len=self.max_lm_lens)

        if zerogaze:
            assert task in ["bottle", "bowl", "car", "chair", "clock", "cup", "fork", "keyboard", 
                            "knife", "laptop", "microwave", "mouse", "oven", "potted plant", "sink", "stop sign", 
                            "toilet", "tv"], "current task is {} must be in {}".format(task, ["bottle", "bottle", "bowl", "car", "chair", "clock", "cup", "fork", "keyboard", "knife", "laptop", "microwave",
                            "mouse", "oven", "potted plant", "sink", "stop sign", "toilet", "tv"])
            fixations_train = list(filter(lambda x: x['task'] != task.replace('_', ' '), fixations_train))
            print("Zero Gaze on", task.replace('_', ' '))

        # 将所有scanpath的第一个注视点设置为图像中心
        # for traj in fixations_train:
        #     traj['tgt_seq_x'][0] = 256
        #     traj['tgt_seq_y'][0] = 160
    
        if condition=="present":  #有2661并没有找到目标，暂时并未清除该部分数据
            bbox_annos_path = "/data/lyt/01-Datasets/01-ScanPath-Datasets/coco_search18/raw/COCOSearch18/dataset_json/bbox_annos.npy"
            bbox_annos = np.load(bbox_annos_path, allow_pickle=True).item()
            self.fixs = cutFixOnTarget(fixations_train, bbox_annos)
        else:
            self.fixs = fixations_train

        # tasks = list(set([x['task'] for x in self.fixs]))
        # tasks = ['car', 'bottle', 'knife', 'bowl', 'sink', 'clock', 'cup', 'toilet', 'mouse', 'laptop', 'stop sign', 'tv', 'microwave', 'fork', 'oven', 'chair', 'potted plant', 'keyboard']
        # tasks_dict1 = {}
        # for task in tasks:
        #     target_fixs = [x for x in self.fixs if x['task']==task]
        #     point_num_0 = sum(len(x['tgt_seq_x']) for x in target_fixs)
        #     point_num_1 = len(target_fixs) * max_lm_lens - point_num_0
        #     # point_num_0 = sum(len(x['tgt_seq_x']) for x in self.fixs if x['task']==task)
        #     # point_num_1 = len(self.fixs) * max_lm_lens - point_num_0
        #     ratio = point_num_0 / point_num_1
        #     tasks_dict1[task] = ratio
        #     # ratio = sum(len(x['tgt_seq_x']) for x in self.fixs) / len(self.fixs) - 1
        #     print("termination pos weight: {:.3f}".format(ratio))
        # print(tasks_dict1)

        if condition=="present":
            self.img_dir = "/data/lyt/01-Datasets/01-ScanPath-Datasets/coco_search18/raw/COCOSearch18/images/"
        else:
            self.img_dir = "/data/lyt/01-Datasets/01-ScanPath-Datasets/coco_search18/raw/COCOSearch18/coco_search18_images_TA/"

        self.begin_str = f'{DEFAULT_IMAGE_TOKEN}\n'
        if extra_image_processor is not None:
            self.extra_image_processor = BUILDER.build(extra_image_processor)
        self.arch_type = arch_type
        if self.arch_type == 'qwen':
            self.IMG_CONTEXT_TOKEN = '<|image_pad|>'
            self.IMG_START_TOKEN = '<|vision_start|>'
            self.IMG_END_TOKEN = '<|vision_end|>'
        elif self.arch_type == 'llava':
            self.IMG_CONTEXT_TOKEN = '<image>'
            self.IMG_START_TOKEN = ''
            self.IMG_END_TOKEN = ''
        
        self.tokenizer = BUILDER.build(tokenizer)
        if special_tokens is not None:
            self.tokenizer.add_tokens(special_tokens, special_tokens=True)

        print(self.tokenizer.special_tokens_map)

        self.template = prompt_template
        self.max_length = max_length
        if self.arch_type == 'intern_vl':
            # self._system = '你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。'
            self._system = ''
            self.template['INSTRUCTION'] = '<|user|>\n{input}<|end|><|assistant|>\n'
        elif self.arch_type == 'qwen':
            self._system = ''
        elif self.arch_type == 'llava':
            self._system = ''

        self.num_classes_per_sample = num_classes_per_sample
        self.image_size = 448
        if self.arch_type == 'llava':
            self.image_size = 336

        if preprocessor is None:
            self.transformer = T.Compose([
                T.Lambda(lambda img: img.convert('RGB')
                if img.mode != 'RGB' else img),
                T.Resize((self.image_size, self.image_size),
                     interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
            ])
            self.preprocessor = None
        else:
            self.transformer = None
            self.preprocessor = BUILDER.build(preprocessor)
        self.arch_type = arch_type
        self.single_image_mode = single_image_mode
        self._max_refetch = 1000

        self.counter=0

    def _parse_annotations(self, fix):
        image_path = join(self.img_dir, fix['task'].replace(' ', '_'), fix['img_name'])
        
        task_with_token = '<|object_ref_start|> ' + fix['task']+ '<|object_ref_end|>'
        # task_with_token = fix['task']

        assert fix['task'] not in ["potted_plant", "stop_sign"], "make sure the category has right name" 

        text = [task_with_token]
        phrases = []
        index = np.random.choice(len(text), self.num_classes_per_sample, replace=True)   #同一个图像可能有多个prompt，为了避免趋向于某一图像，每个图像使用的概率要相同
        for idx in index: 
            phrase = text[idx].lower()
            if '.' == phrase[-1]:
                phrase = phrase[:-1]
            phrases.append(phrase)

        conversation = []
        for i, phrase in enumerate(phrases):
            # question = random.choice(SEG_QUESTIONS).format(class_name=phrase)
            # if i == 0:
            #     question = self.begin_str + question

            # conversation.append({'from': 'human', 'value': question})
            # conversation.append({'from': 'gpt', 'value': random.choice(ANSWER_LIST)})
            
            seg_question = 'Please segment {class_name} in this image.'
            question = seg_question.format(class_name=phrase)
            if i == 0:
                question = self.begin_str + question
            conversation.append({'from': 'human', 'value': question})
            anser1 = "Sure, it is [SEG]."
            conversation.append({'from': 'gpt', 'value': anser1})

        fix.update({
            'conversations': conversation,
            'image_path': image_path
        })
        return fix

    def prepare_data(self, idx):
        fix = self.fixs[idx]
        data_dict = self._parse_annotations(fix)

        out_data_dict = {}

        image_file = data_dict['image_path']

        image = self._read_image(image_file) #1
        if image is None:
            return None
        image = image.resize((520, 320)) #固定到指定大小

        # Process image using base class method
        image_data = self._process_single_image(image, self.single_image_mode)
        out_data_dict.update(image_data)
        
        # Create image token string and get input/labels
        image_token_str = self._create_image_token_string(image_data['num_image_tokens'])
        conversation = self._process_conversations_for_encoding(fix['conversations'], image_token_str)
        token_dict = self.get_inputid_labels(conversation)    
        out_data_dict.update(token_dict)

        #prompt 转化为文字后可视化
        # prompt_seen = self.tokenizer.convert_ids_to_tokens(token_dict['input_ids'])
        # prompt_seen1 = prompt_seen[-50:]
        # token_str = self.tokenizer.decode(token_dict['input_ids'])

        # # 找到object_ref_start和object_ref_end的位置, 然后提取出对应的subtokens
        # start_token_mask = torch.tensor(token_dict['input_ids']) == 151646
        # end_token_mask = torch.tensor(token_dict['input_ids']) == 151647
        # start_token_mask = torch.where(start_token_mask)[0]
        # end_token_mask = torch.where(end_token_mask)[0]
        # start, end = start_token_mask[0], end_token_mask[0] 

        # # tokens = self.tokenizer.tokenize(conversation[0]['input'])
        # tokens = self.tokenizer.convert_ids_to_tokens(token_dict['input_ids'])
        # cleaned_tokens = [token.replace('Ġ', '') for token in tokens] 
        # cleaned_tokens = cleaned_tokens[start+1:end]

        # 确保 fixation tensor 在 CPU 上，避免显存泄漏
        # 使用 clone().detach() 创建新 tensor，并确保在 CPU 上
        fixation_dic = {'fixation_x': fix['tgt_seq_x'].clone().detach().cpu(),
                        'fixation_y': fix['tgt_seq_y'].clone().detach().cpu(),
                        'fixation_t': fix['tgt_seq_t'].clone().detach().cpu(),
                        'task': fix['task']
                        }
        
        out_data_dict.update(fixation_dic)
        return out_data_dict
    
    def _rand_another(self) -> int:
        return np.random.randint(0, len(self.fixs))
    
    def __len__(self):
        return len(self.fixs)
    
    def __getitem__(self, index):
        for _ in range(self._max_refetch + 1):
            data = self.prepare_data(index)
            # Broken images may cause the returned data to be None
            if data is None:
                index = self._rand_another()
                continue
            return data

def get_args_parser_pretrain():
    parser = argparse.ArgumentParser('Gaze Transformer Pretrainer', add_help=False)
    parser.add_argument('--dataset_dir', default= '/data/lyt/01-Datasets/01-ScanPath-Datasets/ART_data/data/', type=str)
    parser.add_argument('--train_file', default= 'refcocogaze_train_correct_tf_512X320_6.json', type=str)
    parser.add_argument('--img_dir', default= '/data/lyt/01-Datasets/01-ScanPath-Datasets/ART_data/data/images_512X320', type=str)

    parser.add_argument('--max_pack_len', default=4, type=int)
    parser.add_argument('--max_lm_lens', default=16, type=int)

    parser.add_argument('--im_h', default=320, type=int, help='image vertical size')
    parser.add_argument('--im_w', default=512, type=int, help='image horizontal size')

    return parser


if __name__=="__main__":
    import argparse
    from transformers import AutoTokenizer
    import numpy as np
    from torchvision.transforms.functional import resize, to_pil_image  # type: ignore

    # Dataset = LN_Dataset(fixs=train_refgazes, img_dir=args.img_dir, tokenizer=tokenizer, extra_image_processor=extra_image_processor, prompt_template = prompt_template, args=args,
    #                             max_length = 8192, num_classes_per_sample=5)
    
    # for i in range(1000):
    #     tmp = Dataset[i]