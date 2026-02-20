from typing import Literal
from collections import OrderedDict
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
                loss_fn_xy = None,
                loss_fn_token = None,
                torch_dtype=torch.bfloat16,
                pretrained_pth=None,
                special_tokens=None,
                arch_type:Literal['intern_vl', 'qwen', 'llava']='intern_vl',
                training_bs:int=0,
                max_predict_lens:int=16,
                condition="present",
                ):
        super().__init__()
        if special_tokens is None:
            special_tokens = ['[SEG]']

        # self.decoder 来融合当前的视觉文本信息和历史的注视点信息
        self.Scanpath_Decoder = BUILDER.build(decoder)     
        self.in_dim = self.Scanpath_Decoder.hidden_dim

        self.max_predict_lens = max_predict_lens 
        self.token_predictor = nn.Sequential(
            nn.Linear(self.in_dim, self.in_dim), 
            nn.LeakyReLU(negative_slope=0.01, inplace=True), 
            nn.Dropout(0.1),
            nn.Linear(self.in_dim, 2) #只分类为0或者1
        )
        self.generator_y_mu = nn.Sequential(
            nn.Linear(self.in_dim, self.in_dim), 
            nn.LeakyReLU(negative_slope=0.01, inplace=True), 
            nn.Dropout(0.1),
            nn.Linear(self.in_dim, 1)
        )
        self.generator_x_mu = nn.Sequential(
            nn.Linear(self.in_dim, self.in_dim), 
            nn.LeakyReLU(negative_slope=0.01, inplace=True), 
            nn.Dropout(0.1),
            nn.Linear(self.in_dim, 1)
        )

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

        self.activation = F.sigmoid

        self.softmax = nn.LogSoftmax(dim=-1)

        self.loss_fn_xy = BUILDER.build(loss_fn_xy)  
        self.loss_fn_token = BUILDER.build(loss_fn_token) 
        
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

        path = "/data/lyt/03-Repositories/01-ours/ScanVLA/ScanHisVLA/work_dirs/ScanHisVLA_AiR/iter_26560.pth"
        pretrained_state_dict = guess_load_checkpoint(path)
        self.load_state_dict(pretrained_state_dict, strict=False)
        del pretrained_state_dict

        # 暂时先不训练
        # self.mllm.use_llm_lora=False
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
             for k, v in state_dict_all.items() if 'token_predictor' in k or 'generator_y_mu' in k or 'generator_x_mu' in k}
        to_return.update(state_dict_predictor)
        
        # 添加decoder的参数
        state_dict_decoder = {k: v for k, v in state_dict_all.items() if 'Scanpath_Decoder' in k}

        to_return.update(state_dict_decoder)

        prefix = kwargs.pop('prefix', '')
        state_dict_mllm = self.mllm.state_dict(*args, prefix=prefix + 'mllm.', **kwargs)
        to_return.update(state_dict_mllm)
        return to_return

    def calc_xyt_token_loss(self, x_mu, y_mu, token_predict, data):
        gt_scanpath_x = data.pop('scanpath_x', None) 
        gt_scanpath_y = data.pop('scanpath_y',None) 

        #如果超出长度，需要截断 
        truncate_len = self.max_predict_lens  
        gt_scanpath_x = gt_scanpath_x[:,:truncate_len]
        gt_scanpath_y = gt_scanpath_y[:,:truncate_len]
        
        gt_scanpath_x_flatten = torch.flatten(gt_scanpath_x)
        gt_scanpath_y_flatten = torch.flatten(gt_scanpath_y)

        no_zero_pos_mask = torch.logical_not(gt_scanpath_x_flatten == 0)
        fixation_cnt = no_zero_pos_mask.sum() + 1e-5

        # 计算 loss
        loss_x = (self.loss_fn_xy(x_mu, gt_scanpath_x_flatten) * no_zero_pos_mask).sum() / fixation_cnt
        loss_y = (self.loss_fn_xy(y_mu, gt_scanpath_y_flatten) * no_zero_pos_mask).sum() / fixation_cnt
        
        # 计算 token loss
        token_gt = (gt_scanpath_x == 0).long() 
        batch_size = gt_scanpath_x.shape[0]

        # a = token_predict
        # b = token_gt.flatten(1).long()
        loss_token = self.loss_fn_token(token_predict,
                                       token_gt.flatten(1).long())
        
        return loss_x, loss_y, loss_token
    
    def forward(self, data, data_samples=None, mode='loss'):
        performance = data['performances'][0] #注意需要bs=1时才成立

        input_ids = data['input_ids']

        output = self.mllm(data, data_samples, mode)
        hidden_states = output.hidden_states[-1]
        del output.hidden_states

        # 查看输入VLM的内容
        ref_encode_see = self.tokenizer.decode(input_ids[0])

        # 找到每个样本中第一个 end_token 的索引
        seg_token_mask = input_ids == self.seg_token_idx
        end_token_indices = torch.argmax(seg_token_mask.int(), dim=1) 
        pred_embeddings_vl = hidden_states[:, end_token_indices, :][0]
        pred_embeddings_vl = pred_embeddings_vl

        vision_start_token_mask = input_ids == self.vision_start_token_idx
        vision_end_token_mask = input_ids == self.vision_end_token_idx
        vision_start_token_indices = torch.argmax(vision_start_token_mask.int(), dim=1)
        vision_end_token_indices = torch.argmax(vision_end_token_mask.int(), dim=1)
        assert vision_start_token_indices[0]==3
        assert vision_end_token_indices[0]==420

        pred_embeddings_vision = torch.stack([
            hidden_states[b, vision_start_token_indices[b]+1:vision_end_token_indices[b], :] for b in range(input_ids.size(0))]).permute(1,0,2)

        outs = self.Scanpath_Decoder(performance = performance, vl_guidance_feats=pred_embeddings_vl, pred_embeddings_vision=pred_embeddings_vision)

        y_mu = self.activation(self.generator_y_mu(outs)).flatten() 
        x_mu = self.activation(self.generator_x_mu(outs)).flatten() 
        token_predict = self.softmax(self.token_predictor(outs)) # [L,B,2]
        token_predict = token_predict.permute(1,2,0)

        loss_x, loss_y, loss_token = self.calc_xyt_token_loss(x_mu, y_mu, token_predict, data)

        llm_loss = output.loss

        loss_dict = {
            'loss_x': loss_x,
            'loss_y': loss_y,
            'loss_token': loss_token,
            'llm_loss': llm_loss,
        }
        return loss_dict

def get_seg_hidden_states(hidden_states, output_ids, seg_id):
    seg_mask = output_ids == seg_id
    n_out = len(seg_mask)
    return hidden_states[-n_out:][seg_mask]


