import copy
import random
import glob
import json
import logging
import os
from typing import Literal

import torch

from mmengine import print_log
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from pycocotools import mask as mask_utils

from xtuner.registry import BUILDER

from os.path import join
DEFAULT_IMAGE_TOKEN = "<image>"

from .base import Sa2VABaseDataset

def get_num_step2target(X, Y, bbox):
    X, Y = np.array(X), np.array(Y)
    on_target_X = np.logical_and(X > bbox[0], X < bbox[0] + bbox[2])
    on_target_Y = np.logical_and(Y > bbox[1], Y < bbox[1] + bbox[3])
    on_target = np.logical_and(on_target_X, on_target_Y)
    if np.sum(on_target) > 0:
        # 返回第一个True的位置
        first_on_target_idx = np.argmax(on_target)
        return first_on_target_idx + 1
    
        # # 返回第一个连续True区间中最后一个True的位置
        # in_true_range = False  # 是否进入第一个True区间
        # last_true_idx = None   # 记录目标索引
    
        # for idx, val in enumerate(on_target):
        #     if val:
        #         in_true_range = True  # 进入True区间
        #         last_true_idx = idx   # 更新最后一个True的索引
        #     else:
        #         if in_true_range:
        #             # 遇到False且已在True区间内 → 第一个True区间结束，直接返回
        #             return last_true_idx + 1
        # return last_true_idx + 1
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
            fixs = "projects/ScanVLA_COCOSearch18/datasets/dataset_json/coco_search18_fixations_TP_train.json"
        elif condition=="absent":
            fixs = "projects/ScanVLA_COCOSearch18/datasets/dataset_json/coco_search18_fixations_TA_train.json"

        fixations_train = json.load(open(fixs, mode='r'))
        fixations_train = fixations2seq(fixations=fixations_train, max_len=self.max_lm_lens)

        if zerogaze:
            assert task in ["bottle", "bowl", "car", "chair", "clock", "cup", "fork", "keyboard", 
                            "knife", "laptop", "microwave", "mouse", "oven", "potted plant", "sink", "stop sign", 
                            "toilet", "tv"], "current task is {} must be in {}".format(task, ["bottle", "bottle", "bowl", "car", "chair", "clock", "cup", "fork", "keyboard", "knife", "laptop", "microwave",
                            "mouse", "oven", "potted plant", "sink", "stop sign", "toilet", "tv"])
            fixations_train = list(filter(lambda x: x['task'] != task.replace('_', ' '), fixations_train))
            print("Zero Gaze on", task.replace('_', ' '))
    
        if condition=="present":  #有2661并没有找到目标，暂时并未清除该部分数据
            bbox_annos_path = "projects/ScanVLA_COCOSearch18/datasets/dataset_json/bbox_annos.npy"
            bbox_annos = np.load(bbox_annos_path, allow_pickle=True).item()
            self.fixs = cutFixOnTarget(fixations_train, bbox_annos)
        else:
            self.fixs = fixations_train

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

    def _parse_annotations(self, fix):
        image_path = join(self.img_dir, fix['task'].replace(' ', '_'), fix['img_name'])
        
        assert fix['task'] not in ["potted_plant", "stop_sign"], "make sure the category has right name" 
        
        # 由于保持VLM完全冻结，我们在这里并不适合添加特殊的tokens
        # task_with_token = '<|object_ref_start|> ' + fix['task']+ '<|object_ref_end|>'
        task_with_token = fix['task']

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

            seg_question = 'Please segment {class_name} in this image.'

            question = seg_question.format(class_name=phrase)
            if i == 0:
                question = self.begin_str + question
            conversation.append({'from': 'human', 'value': question})
            anser1 = "Sure, it is [SEG]."
            # anser1 = "Sure, it is [PATH]."
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


if __name__=="__main__":
    import argparse
    from transformers import AutoTokenizer
    import numpy as np
    from torchvision.transforms.functional import resize, to_pil_image  # type: ignore

    # Dataset = LN_Dataset(fixs=train_refgazes, img_dir=args.img_dir, tokenizer=tokenizer, extra_image_processor=extra_image_processor, prompt_template = prompt_template, args=args,
    #                             max_length = 8192, num_classes_per_sample=5)
    
    # for i in range(1000):
    #     tmp = Dataset[i]