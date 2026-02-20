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

def tokens_to_utterance_indices(tokens, utterance):
    token_indices = [-1] * len(tokens)

    pos_utterance = 0
    length_tokens = 0
    length_utterances = 0
    for i, token in enumerate(tokens):
        if pos_utterance >= len(utterance):
            break
        if ' ' in utterance[pos_utterance]:
            length_utterances += len([char for char in utterance[pos_utterance] if char.isalpha()])
            length_tokens += len(token)
            pos_utterance += 1
            continue
        if token.strip() in set(string.punctuation):
            continue

        if length_tokens >= length_utterances + len([char for char in utterance[pos_utterance] if char.isalpha()]):
            length_utterances += len([char for char in utterance[pos_utterance] if char.isalpha()])
            pos_utterance +=1
            if pos_utterance >= len(utterance):
                break

        if token.lower() == utterance[pos_utterance].lower():
            # 避免因为重复单词在utterance中得上一个位置导致的对齐错误
            if length_tokens < length_utterances:
                length_tokens += len(token)
            else:
                token_indices[i] = pos_utterance
                pos_utterance += 1
                length_tokens += len(token)
                length_utterances += len(token)
        else:
            length_tokens += len(token)
    return token_indices


class LNDataset(Sa2VABaseDataset):
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
    IMG_START_TOKEN = '<img>'
    IMG_END_TOKEN = '</img>'

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, 
                fixs, 
                # img_dir, 
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
        if isinstance(fixs,str) and os.path.exists(fixs):
            train_refgazes = json.load(open(fixs, mode='r'))
            # train_refgazes = pre_process_LN(train_refgazes, max_lm_lens, max_pack_len)  
            self.fixs = train_refgazes
        else:
            self.fixs = fixs 
        self.max_lm_lens = max_lm_lens  #在处理数据时进行截取，每个语言描述中最大的token个数
        self.max_pack_len = max_pack_len #在处理数据时进行截取，每个注视点集合中最大的注视点个数
        
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

        self.img_dir = "/data/lyt/01-Datasets/01-ScanPath-Datasets/coco/"  
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
    
    @property
    def modality_length(self):
        import pickle
        length_list = []
        for idx in range(len(self)):
            length_list.append(100)
        return length_list

    def _parse_annotations(self, fix):
        image_path = fix['image_path']
        
        fix_caption_information = torch.load(fix['record_path'])
        fix.update(fix_caption_information)
        caption_with_token = '<|object_ref_start|> ' + fix['caption']+ '<|object_ref_end|>'

        text = [caption_with_token]

        phrases = []
        index = np.random.choice(len(text), self.num_classes_per_sample, replace=True)   #同一个图像可能有多个prompt，为了避免趋向于某一图像，每个图像使用的概率要相同
        for idx in index: 
            phrase = text[idx].lower()
            if '.' == phrase[-1]:
                phrase = phrase[:-1]
            phrases.append(phrase)

        conversation = []
        for i, phrase in enumerate(phrases):
            if i == 0:
                seg_question = 'Please generate an eye movement scanpath to achieve the following image caption: {class_name}.'
                question = seg_question.format(class_name=phrase)
                question = self.begin_str + question
            else: #以下内容在扫视路径训练时跳过
                question = random.choice(SEG_QUESTIONS).format(class_name=phrase)
                
            conversation.append({'from': 'human', 'value': question})
            conversation.append({'from': 'gpt', 'value': random.choice(ANSWER_LIST)})

        fix.update({
            'conversations': conversation,
            'caption_with_token': caption_with_token,
            'image_path': image_path
        })
        return fix
    
    def get_unpredicted_mask(self, caption, utterance):
        caption_with_token = '<|object_ref_start|> ' + caption + '<|object_ref_end|>' #添加开始和结束token, 避免因为特殊符号导致的编码subtokens改变
        tokens = self.tokenizer.tokenize(caption_with_token)
        cleaned_tokens = [token.replace('Ġ', '') for token in tokens] #取代tokens里的G
        cleaned_tokens = cleaned_tokens[1:-1]  #去除开始和结束符号

        # tokens = self.tokenizer.tokenize(caption)
        # cleaned_tokens = [token.replace('Ġ', '') for token in tokens]  #取代tokens里的G

        cleaned_utterance = [word.strip(string.punctuation) for word in utterance]  #去除每个单词两侧的符号

        # 为每一个clened_tokens中的元素预测其相应元素在clened_utterance中的位置。
        token_utterance_indices = tokens_to_utterance_indices(cleaned_tokens,cleaned_utterance)
        unpredicted_mask = [1 if x==-1 else 0 for x in token_utterance_indices]
        unpredicted_mask = unpredicted_mask[:self.max_lm_lens]

        return unpredicted_mask, token_utterance_indices
    
    def get_unpredicted_mask_with_cleaned_tokens(self, cleaned_tokens, utterance):
        cleaned_utterance = [word.strip(string.punctuation) for word in utterance]  #去除每个单词两侧的符号

        # 为每一个clened_tokens中的元素预测其相应元素在clened_utterance中的位置。
        token_utterance_indices = tokens_to_utterance_indices(cleaned_tokens,cleaned_utterance)
        unpredicted_mask = [1 if x==-1 else 0 for x in token_utterance_indices]
        unpredicted_mask = unpredicted_mask[:self.max_lm_lens]

        return unpredicted_mask, token_utterance_indices
    
    def transform_fixation_by_pos(self, fixation_x, fixation_y, bbox, token_utterance_indices):
        tmp = np.array([-1, -1, -1, -1])
        fixation_x_tokens = [-1 if index == -1 else fixation_x[index] for index in token_utterance_indices]
        fixation_y_tokens = [-1 if index == -1 else fixation_y[index] for index in token_utterance_indices]
        bbox = [tmp if index == -1 else np.array(bbox[index]) for index in token_utterance_indices]
        
        fixation_x_tokens = np.array(fixation_x_tokens[:self.max_lm_lens])
        fixation_y_tokens = np.array(fixation_y_tokens[:self.max_lm_lens])
        bbox = np.array(bbox[:self.max_lm_lens], dtype=object)  # Ensure dtype=object for nested arrays

        return fixation_x_tokens, fixation_y_tokens, bbox

    def update_fixation_and_bbox_by_singular_values(self, fixation_x_tokens, fixation_y_tokens, bbox, unpredicted_mask):
        # 对于有奇异值的位置（位置小于0或者大于1），标记为1
        zero_mask1 = (fixation_x_tokens <= 0) | (fixation_x_tokens >= 1)
        zero_mask2 = (fixation_y_tokens <= 0) | (fixation_y_tokens >= 1)
        zero_mask = np.bitwise_or(zero_mask1, zero_mask2)

        # 对于有奇异值的行全清空，不进行计算
        fixation_x_tokens[zero_mask] = 0
        fixation_y_tokens[zero_mask] = 0
        # bbox[zero_mask] = 0
        # bbox = np.array([np.array([0.0, 0.0, 0.0, 0.0]) if True else bbox[i] for i, mask in enumerate(zero_mask)], dtype=object)
        bbox = np.array([np.array([0.0, 0.0, 0.0, 0.0]) if mask else np.array(bbox[i], dtype=float) for i, mask in enumerate(zero_mask)], dtype=float)

        # 将含奇异值的单词注视点和不能预测注视点的单词注视点的掩码进行合并
        unpredicted_mask = np.array(unpredicted_mask) | zero_mask[:len(unpredicted_mask)]

        return fixation_x_tokens, fixation_y_tokens, bbox, unpredicted_mask.tolist()

    def prepare_data(self, idx):
        fix = self.fixs[str(idx)]
        fix = self._parse_annotations(fix)

        out_data_dict = {}
        image_file = fix['image_path']

        image = self._read_image(image_file) #1
        image = image.resize((520, 320)) #固定到指定大小 

        if image is None:
            return None
        # Process image using base class method
        image_data = self._process_single_image(image, self.single_image_mode)
        out_data_dict.update(image_data)
        
        # Create image token string and get input/labels
        image_token_str = self._create_image_token_string(image_data['num_image_tokens'])
        conversation = self._process_conversations_for_encoding(fix['conversations'], image_token_str)
        token_dict = self.get_inputid_labels(conversation)
        out_data_dict.update(token_dict)


        # 找到object_ref_start和object_ref_end的位置, 然后提取出对应的subtokens
        start_token_mask = torch.tensor(token_dict['input_ids']) == 151646
        end_token_mask = torch.tensor(token_dict['input_ids']) == 151647
        start_token_mask = torch.where(start_token_mask)[0]
        end_token_mask = torch.where(end_token_mask)[0]
        start, end = start_token_mask[0], end_token_mask[0] 

        # tokens = self.tokenizer.tokenize(conversation[0]['input'])
        tokens = self.tokenizer.convert_ids_to_tokens(token_dict['input_ids'])
        cleaned_tokens = [token.replace('Ġ', '') for token in tokens] 
        cleaned_tokens = cleaned_tokens[start+1:end]

        # caption = fix['caption']
        # 为每一个clened_tokens中的元素预测其相应元素在clened_utterance中的位置
        # unpredicted_mask, token_utterance_indices = self.get_unpredicted_mask(fix['caption'], fix['utterance'])
        unpredicted_mask, token_utterance_indices = self.get_unpredicted_mask_with_cleaned_tokens(cleaned_tokens, fix['utterance'])

        # cleaned_tokens_len = len(cleaned_tokens)
        # unpredicted_mask_len = len(unpredicted_mask)
        # if cleaned_tokens_len != unpredicted_mask_len:
        #     encode_seen1 = self.tokenizer(caption, add_special_tokens=False, return_offsets_mapping=True)
        #     word_ids_seen1 = encode_seen1.word_ids()

        #     caption_with_token = '<|object_ref_start|> ' + caption + '<|object_ref_end|>'
        #     encode_seen2 = self.tokenizer(caption_with_token, add_special_tokens=False, return_offsets_mapping=True)
        #     word_ids_seen2 = encode_seen2.word_ids()
        #     print('1')

        # 根据位置将注视点信息进行转换
        fixation_x_tokens, fixation_y_tokens, bbox = self.transform_fixation_by_pos(fix['fixation_x'], fix['fixation_y'], fix['bbox_np'], token_utterance_indices)
        # 根据奇异值的位置对fixation,bbox以及unpredicted_mask进行更新
        fixation_x_tokens, fixation_y_tokens, bbox, unpredicted_mask = self.update_fixation_and_bbox_by_singular_values(fixation_x_tokens, fixation_y_tokens, bbox, unpredicted_mask)

        # # 需要对单词在token化过程中进行的拆分进行考虑
        # start_token_mask = torch.tensor(token_dict['input_ids']) == 151646
        # end_token_mask = torch.tensor(token_dict['input_ids']) == 151647
        # start_token_mask = torch.where(start_token_mask)[0]
        # end_token_mask = torch.where(end_token_mask)[0]
        # start, end = start_token_mask[0], end_token_mask[0] 
        # predicted_len = end - start - 1
        # gt_len = fixation_x_tokens.shape[0]

        if end - start - 1 != len(cleaned_tokens):
            self.counter += 1
            print(f"due to tokenizer's subword tokenization, length mismatch occurs {self.counter} times, Predicted length: {end - start - 1}, Ground Truth length: {len(cleaned_tokens)}")
            return None

        fixation_dic = {'fixation_x': torch.tensor(fixation_x_tokens),
                        'fixation_y': torch.tensor(fixation_y_tokens),
                        'unpredicted_mask': torch.tensor(unpredicted_mask),
                        'bbox': torch.tensor(bbox)
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