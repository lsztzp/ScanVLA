import argparse
import json
import os
from datetime import datetime
from os.path import join

import numpy as np
from torch import nn
from tqdm import tqdm
import argparse
import os.path as osp
from mmengine.dist import (collect_results, get_dist_info, get_rank, init_dist,
                           master_only)
from xtuner.registry import BUILDER
from xtuner.configs import cfgs_name_path
from xtuner.model.utils import guess_load_checkpoint
from mmengine.config import Config
from mmengine.fileio import PetrelBackend, get_file_backend
from mmengine.config import ConfigDict
import os
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import torch

import torch 
from projects.llava_sam2.models.preprocess.image_resize import DirectResize
from projects.llava_sam2.models.utils import dynamic_preprocess 
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from projects.glamm.datasets.utils.utils import SEG_QUESTIONS, ANSWER_LIST
from transformers import StoppingCriteriaList, StoppingCriteria, GenerationConfig

from eval_metrics import get_metrics
from projects.llava_sam2.datasets.RefCOCO_Gaze_Dataset import pos_to_fixation


def parse_args():
    parser = argparse.ArgumentParser(description='toHF script')
    parser.add_argument('--config', default='projects/llava_sam2/configs/sa2va_1b_path.py', help='config file name or path.')
    # parser.add_argument('--config', default='/data/lyt/03-Repositories/02-others/03-MultiModality/Sa2VA/projects/llava_sam2/configs/sa2va_1b.py', help='config file name or path.')
    
    # parser.add_argument('--pthmodel',default='/data/lyt/03-Repositories/01-ours/Speech-Directed/Sa2VA_our/work_dirs/sa2va_1b/iter_1000.pth', help='pth model file')
    parser.add_argument('--pthmodel',default='work_dirs/sa2va_1b/best_checkpoint.pth', help='pth model file')
    
    parser.add_argument('--save-path', type=str, default='./work_dirs/hf_model', help='save folder name')

    parser.add_argument('--img_dir', type=str, default='/data/lyt/01-Datasets/01-ScanPath-Datasets/ART_data/data/images_512X320/', help='save folder name')
    parser.add_argument('--dataset_dir', type=str, default='/data/lyt/01-Datasets/01-ScanPath-Datasets/ART_data/data/', help='save folder name')
    parser.add_argument('--test_file', type=str, default='refcocogaze_test_correct_512X320.json', help='save folder name')
    
    args = parser.parse_args()
    return args

args = parse_args()
cfg = Config.fromfile(args.config)
model = BUILDER.build(cfg.model)
pretrained_state_dict = guess_load_checkpoint(args.pthmodel)

# model._merge_lora() #1
model.load_state_dict(pretrained_state_dict, strict=False)
print(f'Load pretrained weight from {args.pthmodel}')
del pretrained_state_dict
model._merge_lora()
model.mllm.transfer_to_hf = True

model = model.eval().cuda()

class StopWordStoppingCriteria(StoppingCriteria):
    """StopWord stopping criteria."""

    def __init__(self, tokenizer, stop_word):
        self.tokenizer = tokenizer
        self.stop_word = stop_word
        self.length = len(self.stop_word)

    def __call__(self, input_ids, *args, **kwargs) -> bool:
        cur_text = self.tokenizer.decode(input_ids[0])
        cur_text = cur_text.replace('\r', '').replace('\n', '')
        return cur_text[-self.length:] == self.stop_word

def get_stop_criteria(
    tokenizer,
    stop_words=[],
):
    stop_criteria = StoppingCriteriaList()
    for word in stop_words:
        stop_criteria.append(StopWordStoppingCriteria(tokenizer, word))
    return stop_criteria

def preprocess_image(image: torch.Tensor, dtype=torch.bfloat16) -> torch.Tensor:
    image = image / 255.

    img_mean = (0.485, 0.456, 0.406)
    img_std = (0.229, 0.224, 0.225)

    img_mean = torch.tensor(img_mean, dtype=dtype, device=image.device)[:, None, None]
    img_std = torch.tensor(img_std, dtype=dtype, device=image.device)[:, None, None]
    image -= img_mean
    image /= img_std

    return image

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained('./pretrained/InternVL2_5-1B/')
    bot_name = 'BOT'
    stop_words = ['<|im_end|>', '<|endoftext|>']
    stop_criteria = get_stop_criteria(
            tokenizer=tokenizer, stop_words=stop_words)
    
    default_generation_kwargs = dict(
            max_new_tokens=2048,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id, #151645
            pad_token_id=(
                tokenizer.pad_token_id
                if tokenizer.pad_token_id is not None
                else tokenizer.eos_token_id
            ),
        )
    gen_config = GenerationConfig(**default_generation_kwargs)

    extra_image_processor = DirectResize(target_length=1024, )

    min_dynamic_patch = 1
    max_dynamic_patch = 12
    downsample_ratio = 0.5
    image_size = 448
    use_thumbnail = True
    patch_size =14 
    patch_token = int((image_size // patch_size) ** 2 * (downsample_ratio ** 2))
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
    IMG_START_TOKEN = '<img>'
    IMG_END_TOKEN = '</img>'

    transformer = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    VP_START_TOKEN = '<vp>'
    VP_END_TOKEN = '</vp>'

    img_context_token_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
    seg_token_idx = tokenizer.convert_tokens_to_ids('[SEG]')

    test_refgazes = json.load(open(join(args.dataset_dir, args.test_file), mode='r'))

    # res = []
    # for case in test_refgazes:
    #     x = case['FIX_X']
    #     y = case['FIX_Y']
    #     index = case['FIX_WORDINDEX']

    #     a,b = [],[]    

    #     i = 0 
    #     tmp_x,tmp_y = [],[]
    #     while i<len(x):
            
    #         if i>0:
    #             if index[i] == index[i-1]:
    #                 tmp_x.append(x[i])
    #                 tmp_y.append(y[i])
    #             else:
    #                 # else:
    #                 a.append(tmp_x)
    #                 b.append(tmp_y)
    #                 tmp_x,tmp_y = [],[]
    #                 tmp_x.append(x[i])
    #                 tmp_y.append(y[i])
    #         else:
    #             tmp_x.append(x[i])
    #             tmp_y.append(y[i])
    #         i+=1
    #     a.append(tmp_x)
    #     b.append(tmp_y)

    #     res.append({'IMAGEFILE': case['IMAGEFILE'], 'X': a, 'Y': b, 'TEXT': case['REF_SENTENCE'], 'BBOX':case['BBOX']})
    # save_path = '/data/lyt/02-Results/01-ScanPath/ScanLLM/Qualititive/huaman.pt'
    # torch.save(res, save_path)

    
    test_refs = []
    test_ref_set = set()
    for case in test_refgazes:
        if case['REF_ID'] not in test_ref_set:
            test_ref_set.add(case['REF_ID'])
            test_refs.append({'REF_ID': case['REF_ID'], 'IMAGEFILE': case['IMAGEFILE'], 'REF_WORDS': case['REF_WORDS'], 'REF_SENTENCE': case['REF_SENTENCE'], 'box': case['BBOX']})

    patch_token = 256

    res = []
    for ref in tqdm(test_refs):
        ref['TEXT'] =  ref['REF_SENTENCE'] #1

        Image_path = join(args.img_dir, ref['IMAGEFILE'])
        image = Image.open(Image_path).convert('RGB')

        ref['REF_SENTENCE'] = '<|object_ref_start|> ' + ref['REF_SENTENCE'] + '<|object_ref_end|>'

        ref_encode_see = tokenizer.tokenize(ref['REF_SENTENCE'], add_special_tokens=False)
        ref_encode = tokenizer(ref['REF_SENTENCE'], add_special_tokens=False, return_offsets_mapping=True)
        ref_offset = ref_encode['offset_mapping']
        ref_offset_mask = pos_to_fixation(ref_offset, ref['REF_SENTENCE'])
        ref_offset_mask = torch.tensor(ref_offset_mask).cuda()

        # if 0 in ref_offset_mask:
        #     continue
        input_dict = {}

        g_image = np.array(image)
        g_image = extra_image_processor.apply_image(g_image)
        g_pixel_values = torch.from_numpy(g_image).permute(2, 0, 1).contiguous().to(torch.bfloat16)
        extra_pixel_values = [g_pixel_values]
        g_pixel_values = torch.stack([
                preprocess_image(pixel) for pixel in extra_pixel_values
            ]).to(torch.bfloat16)

        images = dynamic_preprocess(image, min_dynamic_patch,
                                        max_dynamic_patch,
                                        image_size, use_thumbnail)
        input_dict['vp_overall_mask'] = None
        

        pixel_values = [transformer(image) for image in images]
        pixel_values = torch.stack(pixel_values).to(torch.bfloat16)
        num_image_tokens = pixel_values.shape[0] * patch_token
        num_frames = 1

        input_dict['g_pixel_values'] = g_pixel_values
        input_dict['pixel_values'] = pixel_values

        vp_token_str = ''

        image_token_str = f'{IMG_START_TOKEN}' \
                        f'{IMG_CONTEXT_TOKEN * num_image_tokens}' \
                        f'{IMG_END_TOKEN}'
        image_token_str = image_token_str + '\n'
        image_token_str = image_token_str * num_frames
        image_token_str = image_token_str.strip()
        
        ret_masks = []

        text = "<image>" + 'Please segment {class_name} in this image.'.format(class_name=ref['REF_SENTENCE'])

        text = text.replace('<image>', image_token_str + vp_token_str)
        input_text = ''
        template = '<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n'
        input_text += template.format(
                input=text, round=1, bot_name=bot_name)
        ids = tokenizer.encode(input_text)
        ids = torch.tensor(ids).cuda().unsqueeze(0)

        attention_mask = torch.ones_like(ids, dtype=torch.bool)

        # input_dict['pixel_values'] =torch.stack([input_dict['pixel_values'],input_dict['pixel_values']],dim=0)
        # ids = ids.repeat(2,1)
        # attention_mask = attention_mask.repeat(2,1)
        input_dict['pixel_values'] =torch.stack([input_dict['pixel_values']],dim=0)
        ids = ids.repeat(1,1)
        attention_mask = attention_mask.repeat(1,1)


        bs, seq_len = ids.shape
        position_ids = torch.arange(seq_len).unsqueeze(0).long().repeat(bs, 1)

        mm_inputs = {
                'pixel_values': input_dict['pixel_values'].cuda(),
                'input_ids': ids.cuda(),
                'attention_mask': attention_mask.cuda(),
                'position_ids': position_ids.cuda(),
                'past_key_values': None,
                'labels': None,
                'prompt_masks': None,
                'vp_overall_mask': input_dict['vp_overall_mask'],
            }

        with torch.no_grad():
            generate_output = model.mllm(mm_inputs, None, 'loss')

        input_ids = mm_inputs['input_ids']
        start_token_mask = input_ids == model.start_token_idx
        end_token_mask = input_ids == model.end_token_idx
        
        start_token_mask = torch.where(start_token_mask[0])[0]
        end_token_mask = torch.where(end_token_mask[0])[0]

        hidden_states = generate_output .hidden_states
        # hidden_states = self.text_hidden_fcs(hidden_states[-1])  #1
        hidden_states = hidden_states[-1]  #1

        start, end = start_token_mask[0], end_token_mask[0] 
        pred_embeddings = hidden_states[:, start: end+1, :]

        y_mu, x_mu = model.generator_y_mu(pred_embeddings), model.generator_x_mu(pred_embeddings)
        # tmp = self.token_predictor(pred_embeddings)
        token_predict = model.token_predictor(pred_embeddings).view(pred_embeddings.shape[0],pred_embeddings.shape[1],-1,2)

        token_states = torch.argmax(token_predict, dim=-1)
        # tmp = ref_offset_mask.unsqueeze(0)
        # .expand_as(x_mu)
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
        # res.append({'REF_ID': ref['REF_ID'], 'X': a, 'Y': b , 'TERMINATIONS': -1, 'REPEAT_ID': 0})
        
        res.append({'IMAGEFILE': ref['IMAGEFILE'], 'X': a, 'Y': b, 'TEXT': ref['TEXT'], 'BBOX':ref['box'] })
    # score = get_metrics(res)
    # print(score)
    print('done')
    save_path = '/data/lyt/02-Results/01-ScanPath/ScanLLM/Qualititive/ours.pt'
    torch.save(res, save_path)


print('done!')
# x = [[255.01210021972656], [232.01780700683594], [318.61968994140625], [389.07366943359375, 395.53857421875]]
# y = [[159.66329956054688], [162.040771484375], [135.76873779296875], [110.5812759399414, 105.73650360107422]]
# image = '34037.jpg'
# text = '<|object_ref_start|> silver benz <|object_ref_end|>'












