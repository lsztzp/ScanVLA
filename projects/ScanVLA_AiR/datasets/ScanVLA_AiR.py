import copy
import random
import glob
import json
import logging
import os
from typing import Literal

import torch

from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from pycocotools.coco import COCO

from xtuner.registry import BUILDER

from os.path import join

DEFAULT_IMAGE_TOKEN = "<image>"

from .base import Sa2VABaseDataset


class AiRDataset(Sa2VABaseDataset):
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
                max_predict_lens=16,
                repeats:int = 1,
                name: str = 'RefCOCO_Gaze',
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

        self.img_dir = img_dir
        self.AiR_fixations_dir = fixs
        self.AiR_fixations_file = join(self.AiR_fixations_dir,  "AiR_fixations_train.json")
        
        with open(self.AiR_fixations_file) as json_file:
            self.fixs = json.load(json_file)

        self.max_predict_lens=max_predict_lens

        # 以下部分保持不变
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
        img_name = fix["image_id"]
        image_path = join(self.img_dir, img_name)

        img_size = torch.Tensor([fix['height'], fix['width']])
        
        pos_x = np.array(fix["X"]).astype(np.float32)
        pos_y = np.array(fix["Y"]).astype(np.float32)
        gt_fixation = np.vstack((pos_y, pos_x)).T

        # adapt to free-view
        gt_fixations_all = torch.zeros(1, self.max_predict_lens, 2)
        valid_lens = torch.zeros(1).long()
        valid_lens[0] = len(gt_fixation)

        if len(gt_fixation) <= self.max_predict_lens:
            gt_fixations_all[0][:len(gt_fixation)] = torch.from_numpy(gt_fixation.astype(float))
        else:
            gt_fixations_all[0] = torch.from_numpy(gt_fixation[:self.max_predict_lens].astype(float))

        gt_fixations_all[:, :, 0] /= img_size[0]
        gt_fixations_all[:, :, 1] /= img_size[1]

        performance = fix["subject_answer"] == fix["answer"] and fix["subject_answer"] != "faild"
        question_id = fix["question_id"]
        
        question_txt = fix["question"]
        question_with_token = question_txt

        # anser_txt = fix["fullAnswer"]
        # question_with_token = "Question: " + question_txt + " Answer: " +anser_txt

        text = [question_with_token]
        phrases = []
        index = np.random.choice(len(text), self.num_classes_per_sample, replace=True)   #同一个图像可能有多个prompt，为了避免趋向于某一图像，每个图像使用的概率要相同
        for idx in index: 
            phrase = text[idx].lower()
            if '.' == phrase[-1]:
                phrase = phrase[:-1]
            if '?' == phrase[-1]:
                phrase = phrase[:-1]    
            phrases.append(phrase)

        conversation = []
        for i, phrase in enumerate(phrases):
            # seg_question = 'Please segment {class_name} in this image.'
            # question = seg_question.format(class_name=phrase)
            # vqa_question = "Generate a human gaze trajectory for answering the following question: {class_name}?"

            # if performance:
            #     vqa_question = "Given the visual image and the corresponding question, generate a human scanpath (gaze trajectory) that reflects the cognitive process when correctly answering the following VQA question: {class_name}? The scanpath should first focus on the target region in the image, then verify the target's attributes/location to confirm the correct answer, with gaze points distributed in chronological order of human visual search."
            # else:
            #     vqa_question = "Given the visual image and the corresponding question, generate a human scanpath (gaze trajectory) that reflects the cognitive process when incorrectly answering the following VQA question: {class_name}? The scanpath should show mislocalization of the target or misjudgment of the target's attributes, with gaze points distributed in chronological order of human erroneous visual search."
            
            # if performance:
            #     vqa_question = "Given the visual image and the corresponding question, generate a human scanpath (gaze trajectory) that reflects the cognitive process of correctly answering the following VQA question: {class_name}? The scanpath should first focus on the target region in the image, then verify the target's attributes and location to confirm the correct answer, with gaze points ordered chronologically according to human visual search behavior."
            # else:
            #     vqa_question = "Given the visual image and the corresponding question, generate a human scanpath (gaze trajectory) that reflects the cognitive process of incorrectly answering the following VQA question: {class_name}? The scanpath should exhibit target mislocalization or attribute misjudgment, with gaze points ordered chronologically according to human erroneous visual search behavior."

            if performance:
                vqa_question = "Given the visual image and the corresponding question, generate a human scanpath (gaze trajectory) that reflects the cognitive process of correctly answering the following VQA question: {class_name}?"
            else:
                vqa_question = "Given the visual image and the corresponding question, generate a human scanpath (gaze trajectory) that reflects the cognitive process of incorrectly answering the following VQA question: {class_name}?"

            question = vqa_question.format(class_name=phrase)
            if i == 0:
                question = self.begin_str + question
            conversation.append({'from': 'human', 'value': question})
            anser1 = "Sure, it is [SEG]."
            conversation.append({'from': 'gpt', 'value': anser1})

        fix.update({
            'conversations': conversation,
            'image_path': image_path,
            'question_id': question_id,
            'performance': performance,
            'fixation_x': gt_fixations_all[:, :, 1].squeeze(),
            'fixation_y': gt_fixations_all[:, :, 0].squeeze()
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
        fixation_dic = {
                'performance': fix['performance'],
                'fixation_x': fix['fixation_x'].clone().detach().cpu(),
                'fixation_y': fix['fixation_y'].clone().detach().cpu(),
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

    class DirectResize:
        def __init__(self, target_length: int) -> None:
            self.target_length = target_length

        def apply_image(self, image: np.ndarray) -> np.ndarray:
            """
            Expects a numpy array with shape HxWxC in uint8 format.
            """
            img = to_pil_image(image, mode='RGB')
            return np.array(img.resize((self.target_length, self.target_length)))

    parser = argparse.ArgumentParser('Referral Core PreTrain', parents=[get_args_parser_pretrain()])
    args = parser.parse_args()

    train_refgazes = json.load(open(join(args.dataset_dir, args.train_file), mode='r'))
    train_refgazes = [x for x in train_refgazes if x['NEXT_WORD']=='<pad>']
    train_refgazes = pre_process(train_refgazes, args.max_lm_lens, args.max_pack_len)  
    
    path = './pretrained/InternVL2_5-1B/'
    tokenizer = dict(type=AutoTokenizer.from_pretrained,
        pretrained_model_name_or_path=path)
    extra_image_processor = dict(
        type=DirectResize,
        target_length=1024,
    )

    Dataset = LN_Dataset(fixs=train_refgazes, img_dir=args.img_dir, tokenizer=tokenizer, extra_image_processor=extra_image_processor, prompt_template = prompt_template, args=args,
                                max_length = 8192, num_classes_per_sample=5)
    
    for i in range(1000):
        tmp = Dataset[i]