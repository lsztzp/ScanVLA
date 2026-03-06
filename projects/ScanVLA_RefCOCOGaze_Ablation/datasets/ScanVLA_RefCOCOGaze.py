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
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from xtuner.registry import BUILDER

from os.path import join
import torchvision.transforms.functional as F

from .common import SEG_QUESTIONS, ANSWER_LIST

# from projects.glamm.utils import DEFAULT_IMAGE_TOKEN
DEFAULT_IMAGE_TOKEN = "<image>"

from .base import Sa2VABaseDataset


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


def pre_process(refgazes, max_lm_length=32, max_pack_length=4):
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


class ReferGazeDataset(Sa2VABaseDataset):
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
            train_refgazes = [x for x in train_refgazes if x['NEXT_WORD']=='<pad>']
            train_refgazes = pre_process(train_refgazes, max_lm_lens, max_pack_len)  
            self.fixs = train_refgazes
        else:
            self.fixs = fixs 
        
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

        # print(self.tokenizer.special_tokens_map)

        self.img_dir = img_dir
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
        self.min_dynamic_patch = 1
        self.max_dynamic_patch = 12
        self.downsample_ratio = 0.5
        if self.arch_type == 'llava':
            self.downsample_ratio = 1
        self.image_size = 448
        if self.arch_type == 'llava':
            self.image_size = 336
        self.use_thumbnail = True
        patch_size = 14
        self.patch_token = int((self.image_size // patch_size) ** 2 * (self.downsample_ratio ** 2))

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
    
    @property
    def modality_length(self):
        import pickle
        length_list = []
        for idx in range(len(self)):
            length_list.append(100)
        return length_list

    def _parse_annotations(self, fix):
        image_path = join(self.img_dir, fix['IMAGEFILE'])
        
        text = [fix['REF_SENTENCE']]
        phrases = []
        index = np.random.choice(len(text), self.num_classes_per_sample, replace=True)   #同一个图像可能有多个prompt，为了避免趋向于某一图像，每个图像使用的概率要相同
        for idx in index: 
            phrase = text[idx].lower()
            if '.' == phrase[-1]:
                phrase = phrase[:-1]
            phrases.append(phrase)

        conversation = []
        for i, phrase in enumerate(phrases):

            # 原始的SaVA使用了多个不同的Prompt，我们简化了这一步骤，针对轨迹预测我们仅仅使用单一的prompt.
            if i == 0:
                seg_question = 'Please generate an eye movement scanpath to achieve the following objectives: {class_name}.'
                question = seg_question.format(class_name=phrase)
                question = self.begin_str + question
            else: #以下内容在扫视路径训练时跳过
                question = random.choice(SEG_QUESTIONS).format(class_name=phrase)
                
            conversation.append({'from': 'human', 'value': question})
            conversation.append({'from': 'gpt', 'value': random.choice(ANSWER_LIST)})
            # ANSWER = 'Sure, it is [SEG].'  #1
            # conversation.append({'from': 'gpt', 'value': ANSWER})  #1

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
        # Process image using base class method
        image_data = self._process_single_image(image, self.single_image_mode)
        out_data_dict.update(image_data)
        
        # Create image token string and get input/labels
        image_token_str = self._create_image_token_string(image_data['num_image_tokens'])
        conversation = self._process_conversations_for_encoding(data_dict['conversations'], image_token_str)
        token_dict = self.get_inputid_labels(conversation)
        # token_dict = self.get_inputid_labels(data_dict['conversations'], image_token_str)
        out_data_dict.update(token_dict)

        ref_sentence = fix['REF_SENTENCE']
        ref_encode_see = self.tokenizer.tokenize(ref_sentence, add_special_tokens=False)
        ref_encode = self.tokenizer(ref_sentence, add_special_tokens=False, return_offsets_mapping=True)
        ref_offset = ref_encode['offset_mapping']

        # tokens = self.tokenizer.convert_ids_to_tokens(ref_encode.input_ids)  #得到映射
        word_ids = ref_encode.word_ids()

        ref_offset_mask = pos_to_fixation(ref_offset, ref_sentence)

        CONTEXT_X = torch.tensor([256] + fix['CONTEXT_X'])
        CONTEXT_Y = torch.tensor([160] + fix['CONTEXT_Y'])
        CONTEXT_PACK = torch.tensor([-1] + fix['CONTEXT_PACK']) + 1
        CONTEXT_ORDER = torch.tensor([0] + fix['CONTEXT_ORDER'])
        FIXATION_HIS = torch.stack([CONTEXT_X, CONTEXT_Y, CONTEXT_PACK, CONTEXT_ORDER], dim=0).T

        fixation_dic = {'fixation_x': torch.tensor(fix['fixation_x']),
                        'fixation_y': torch.tensor(fix['fixation_y']),
                        'ref_encode': torch.tensor(ref_encode['input_ids']),
                        'ref_offset_mask': torch.tensor(ref_offset_mask),
                        'fixation_his': FIXATION_HIS,
                        'word_ids': torch.tensor(word_ids)
                        }
        
        out_data_dict.update(fixation_dic)

        return out_data_dict
    
    def _rand_another(self) -> int:
        return np.random.randint(0, len(self.data))
    
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

    Dataset = ReferGazeDataset(fixs=train_refgazes, img_dir=args.img_dir, tokenizer=tokenizer, extra_image_processor=extra_image_processor, prompt_template = prompt_template, args=args,
                                max_length = 8192, num_classes_per_sample=5)
    
    for i in range(1000):
        tmp = Dataset[i]