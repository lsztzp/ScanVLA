from typing import Literal
from collections import OrderedDict
# from pycocotools import mask as _mask

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmengine.model import BaseModel
from xtuner.registry import BUILDER
from xtuner.model.utils import guess_load_checkpoint

from peft import PeftModelForCausalLM

from transformers import AutoImageProcessor, AutoVideoProcessor

class ScanVLAModel(BaseModel):
    def __init__(self,
                 mllm,
                 tokenizer,
                 decoder,
                 torch_dtype=torch.bfloat16,
                 pretrained_pth=None,
                 special_tokens=None,
                 # for arch selection
                 arch_type:Literal['intern_vl', 'qwen', 'llava']='intern_vl',
                 ):
        super().__init__()
        if special_tokens is None:
            special_tokens = ['[SEG]']
        
        self.Scanpath_Decoder = BUILDER.build(decoder)     
        self.in_dim = self.Scanpath_Decoder.hidden_dim

        # 预测注视点的box, 预测box的四个坐标，以及中心点的x,y坐标，前4个为bbox，后两个为中心点
        self.point_predictor = nn.Sequential(
            nn.Linear(self.in_dim, self.in_dim), 
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(0.05),
            nn.Linear(self.in_dim, 1), 
        )
        self.point_activation = F.sigmoid

        self.mllm = BUILDER.build(mllm)
        self.arch_type = arch_type

        tokenizer = BUILDER.build(tokenizer)
        self._add_special_tokens(tokenizer, special_tokens)

        self.tokenizer = tokenizer

        if arch_type == 'qwen':
            image_processor = AutoImageProcessor.from_pretrained(mllm['model_path'], trust_remote_code=True)
            video_processor = AutoVideoProcessor.from_pretrained(mllm['model_path'], trust_remote_code=True)
            self.mllm._init_processor(image_processor, video_processor)

        # FIX: Untie weights for Qwen model
        if self.arch_type == 'qwen' and self.mllm.model.config.tie_word_embeddings:
            print("Untying embed_tokens and lm_head weights for Qwen model.")
            self.mllm.model.config.tie_word_embeddings = False
            lm_head = self.mllm.model.get_output_embeddings()
            if lm_head is not None:
                input_embeddings = self.mllm.model.get_input_embeddings()
                lm_head.weight = nn.Parameter(input_embeddings.weight.clone())

        self.torch_dtype = torch_dtype

        if pretrained_pth is not None:
            pretrained_state_dict = guess_load_checkpoint(pretrained_pth)
            self.load_state_dict(pretrained_state_dict, strict=False)
            print(f'Load pretrained weight from {pretrained_pth}')

            # FIX: Force update lm_head weight after loading state_dict
            if self.arch_type == 'qwen':
                print("Force updating lm_head weight from pretrained state_dict.")
                lm_head_key = 'mllm.model.lm_head.weight'
                if lm_head_key in pretrained_state_dict:
                    lm_head_weight = pretrained_state_dict[lm_head_key]
                    self.mllm.model.get_output_embeddings().weight.data.copy_(lm_head_weight)
                    print(f"Successfully updated lm_head weight from key: {lm_head_key}")
                else:
                    print(f"Warning: lm_head weight key '{lm_head_key}' not found in pretrained_state_dict.")

            del pretrained_state_dict

        # path = "/data/lyt/03-Repositories/01-ours/ScanVLA/ScaHisVLA/work_dirs/scanhisvla_qwen3_2b_LN/iter_18000.pth"
        # pretrained_state_dict = guess_load_checkpoint(path)
        # self.load_state_dict(pretrained_state_dict, strict=False)
        # del pretrained_state_dict

        if self.mllm.use_llm_lora:
            self.mllm.manual_prepare_llm_for_lora()

        # Print gradient status of all weights in self.mllm.model.base_model.model
        print("\n" + "="*80)
        print("GRADIENT STATUS OF MLLM.MODEL WEIGHTS")
        print("="*80)
        
        try:
            base_model = self.mllm.model
            total_params = 0
            trainable_params = 0
            
            for name, param in base_model.named_parameters():
                total_params += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
                    grad_status = "✓ TRAINABLE"
                else:
                    grad_status = "✗ FROZEN"
                
                print(f"{name:<60} | {grad_status} | Shape: {tuple(param.shape)} | Params: {param.numel():,}")
            
            print("-" * 80)
            print(f"SUMMARY:")
            print(f"  Total parameters: {total_params:,}")
            print(f"  Trainable parameters: {trainable_params:,}")
            print(f"  Frozen parameters: {total_params - trainable_params:,}")
            print(f"  Trainable ratio: {trainable_params/total_params*100:.2f}%")
            print("=" * 80)
            
        except Exception as e:
            print(f"Failed to access self.mllm.model: {e}")
            print("Available attributes in self.mllm.model:")
            print([attr for attr in dir(self.mllm.model) if not attr.startswith('_')])

        self.dtype = torch.bfloat16
        self.to(self.dtype)


    def _add_special_tokens(self, tokenizer, special_tokens):
        self.mllm.add_special_tokens(tokenizer, special_tokens)
        self.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0] # required to make add_special_tokens to be False to avoid <bos> or <eos>
        self.start_token_idx = tokenizer("<|object_ref_start|>", add_special_tokens=False).input_ids[0] #1
        self.end_token_idx = tokenizer("<|object_ref_end|>", add_special_tokens=False).input_ids[0] #1
        self.vision_start_token_idx = tokenizer("<|vision_start|>", add_special_tokens=False).input_ids[0] #1
        self.vision_end_token_idx = tokenizer("<|vision_end|>", add_special_tokens=False).input_ids[0] #1

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        return super().load_state_dict(state_dict, strict, assign)

    def _merge_lora(self):
        if isinstance(self.mllm.model, PeftModelForCausalLM):
            self.mllm.model = self.mllm.model.merge_and_unload()
            return
        
        try:
            self.mllm.model.language_model = self.mllm.model.language_model.merge_and_unload()
        except:
            print("Skip language model, no LoRA in it !!!")
        try:
            self.mllm.model.vision_model = self.mllm.model.vision_model.merge_and_unload()
        except:
            print("Skip vision encoder, no LoRA in it !!!")
        return

    def all_state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        return state_dict

    def state_dict(self, *args, **kwargs):
        state_dict_all = self.all_state_dict(*args, **kwargs)

        to_return = OrderedDict()

        # 添加scanpath预测头的参数
        state_dict_predictor = {k: v
             for k, v in state_dict_all.items() if 'point_predictor' in k }
        to_return.update(state_dict_predictor)
        
        # 添加decoder的参数
        state_dict_decoder = {k: v for k, v in state_dict_all.items() if 'Scanpath_Decoder' in k}

        to_return.update(state_dict_decoder)

        prefix = kwargs.pop('prefix', '')
        state_dict_mllm = self.mllm.state_dict(*args, prefix=prefix + 'mllm.', **kwargs)
        to_return.update(state_dict_mllm)
        return to_return
    
    def point_loss_fn(self, point_outputs, gt_bboxes, fixation_x, fixation_y, unpredicted_mask):
        """
        point_outputs: (B, L, 6) 预测的注视点信息，前4个为bbox，后两个为中心点
        gt_bboxes: (B, L, 4) 真实的bbox信息
        fixation_x: (B, L) 真实的注视点x坐标
        fixation_y: (B, L) 真实的注视点y坐标
        unpredicted_mask: (B, L) 标记哪些位置需要计算损失
        """
        pred_bboxes = point_outputs[:, :, :4]
        pred_center_x = point_outputs[:, :, 4]
        pred_center_y = point_outputs[:, :, 5]

        # 计算bbox的L1损失
        loss_bbox = F.l1_loss(pred_bboxes, gt_bboxes, reduction='none')  # (B, L, 4)
        loss_bbox = loss_bbox.mean(dim=-1)  # (B, L)

        # 计算中心点的L1损失
        loss_center_x = F.l1_loss(pred_center_x, fixation_x, reduction='none')  # (B, L)
        loss_center_y = F.l1_loss(pred_center_y, fixation_y, reduction='none')  # (B, L)

        loss_center = loss_center_x + loss_center_y  # (B, L)

        # 综合损失
        total_loss = loss_bbox + loss_center  # (B, L)

        # 只计算unpredicted_mask为0的位置
        predicted_mask = 1 - unpredicted_mask  # 将unpredicted_mask反转，1表示需要计算损失的位置
        total_loss = total_loss * predicted_mask  # (B, L)
        loss = total_loss.sum() / (predicted_mask.sum() + 1e-5)

        return loss

    def forward(self, data, data_samples=None, mode='loss'):
        input_ids = data['input_ids']

        output = self.mllm(data, data_samples, mode)

        hidden_states = output.hidden_states
        hidden_states = hidden_states[-1]

        ref_encode_see = self.tokenizer.decode(input_ids[0])
        # 找到视觉特征的嵌入位置
        vision_start_token_mask = input_ids == self.vision_start_token_idx
        vision_end_token_mask = input_ids == self.vision_end_token_idx
        vision_start_token_indices = torch.argmax(vision_start_token_mask.int(), dim=1)
        vision_end_token_indices = torch.argmax(vision_end_token_mask.int(), dim=1)
        pred_embeddings_vision = torch.stack([
            hidden_states[b, vision_start_token_indices[b]+1:vision_end_token_indices[b], :] for b in range(input_ids.size(0))]).permute(1,0,2)
        
        # 找到ref的开始和结束位置
        ref_start_token_mask = input_ids == self.start_token_idx
        ref_end_token_mask = input_ids == self.end_token_idx
        ref_start_token_indices = torch.argmax(ref_start_token_mask.int(), dim=1)
        ref_end_token_indices = torch.argmax(ref_end_token_mask.int(), dim=1)
        pred_embeddings_vl = torch.stack([
            hidden_states[b, ref_start_token_indices[b]+1:ref_end_token_indices[b], :] for b in range(input_ids.size(0))])
        pred_embeddings_vl = pred_embeddings_vl[0] #需要保证bs=1

        outs = self.Scanpath_Decoder(vl_guidance_feats=pred_embeddings_vl, pred_embeddings_vision=pred_embeddings_vision)
        
        point_outputs = self.point_predictor(outs).permute(2,1,0) #(B,L,6)
        point_outputs = self.point_activation(point_outputs)  

        point_loss = self.point_loss_fn(point_outputs, data['bbox'], data['scanpath_x'], data['scanpath_y'], data['unpredicted_mask'])
        
        loss_dict = {
            'point_loss': point_loss,
            'llm_loss': output.loss,
        }
        return loss_dict


