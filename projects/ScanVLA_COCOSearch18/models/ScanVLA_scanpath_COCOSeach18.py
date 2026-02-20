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
                loss_fn_t = None,
                loss_fn_token = None,
                torch_dtype=torch.bfloat16,
                pretrained_pth=None,
                special_tokens=None,
                arch_type:Literal['intern_vl', 'qwen', 'llava']='intern_vl',
                training_bs:int=0,
                max_predict_lens:int=7,
                condition="present",
                ):
        super().__init__()
        if special_tokens is None:
            special_tokens = ['[SEG]']

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
        self.generator_t_mu = nn.Sequential(
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

        self.activation = F.relu
        self.softmax = nn.LogSoftmax(dim=-1)

        self.loss_fn_xy = BUILDER.build(loss_fn_xy)  #1 
        self.loss_fn_t = BUILDER.build(loss_fn_t)

        # self.token_ratio_TP = {'car': 0.7663286004056795, 'bottle': 0.8493525896414342, 'knife': 0.7293643945930399, 
        #                     'bowl': 0.8549450549450549, 'sink': 0.6337963669668752, 'clock': 0.5535815002820079, 
        #                     'cup': 0.7237414107418314, 'toilet': 0.5339830084957521, 'mouse': 0.6720329464619993, 
        #                     'laptop': 0.6987704918032787, 'stop sign': 0.5141521682679824, 'tv': 0.5745679012345679, 
        #                     'microwave': 0.7616121897272057, 'fork': 0.6759142496847415, 'oven': 0.6866515837104072, 
        #                     'chair': 0.794027149321267, 'potted plant': 0.8693181818181818, 'keyboard': 0.5279456193353474}
        # self.token_ratio_TA = {'car': 2.7416481069042318, 'bottle': 2.194335169158143, 'knife': 2.708108108108108, 
        #                        'bowl': 3.44300518134715, 'sink': 1.7278177458033572, 'clock': 2.5731857318573184, 
        #                        'cup': 2.5627637130801686, 'toilet': 1.048417132216015, 'mouse': 2.1742243436754176, 
        #                        'laptop': 2.6440677966101696, 'stop sign': 3.039344262295082, 'tv': 1.4150677697588452, 
        #                        'microwave': 3.810844892812106, 'fork': 2.0691721132897603, 'oven': 2.510028653295129, 
        #                        'chair': 2.14013709063214, 'potted plant': 2.9132706374085684, 'keyboard': 1.732540408661177}
        # self.loss_fn_token_dict = {}
        # assert condition in ('present', 'absent')
        # if condition == "present":
        #     for cls_name, tp_ratio in self.token_ratio_TP.items():
        #         weights = torch.tensor([1, tp_ratio])
        #         self.loss_fn_token_dict[cls_name] = torch.nn.NLLLoss(weight=weights)
        # else:
        #     for cls_name, tp_ratio in self.token_ratio_TA.items():
        #         weights = torch.tensor([1, tp_ratio])
        #         self.loss_fn_token_dict[cls_name] = torch.nn.NLLLoss(weight=weights)
        
        # if condition == "present":
        #     weights = torch.tensor([1.0, 0.679578355942])
        # elif condition == "absent":
        #     weights = torch.tensor([1.0, 2.1681943746860872])
        # self.loss_fn_token = torch.nn.NLLLoss(weight=weights)

        self.loss_fn_token = BUILDER.build(loss_fn_token) #1
        
        # self.torch_dtype = torch_dtype

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

        # 如果需要断点重新训练，打开以下代码
        # path = "/data/lyt/03-Repositories/01-ours/ScanVLA/ScaHisVLA/work_dirs/ScanVLA_TA_class_ratio/iter_21632.pth"
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
             for k, v in state_dict_all.items() if 'token_predictor' in k or 'generator_y_mu' in k or 'generator_x_mu' in k or 'generator_t_mu' in k}
        to_return.update(state_dict_predictor)
        
        # 添加decoder的参数
        state_dict_decoder = {k: v for k, v in state_dict_all.items() if 'Scanpath_Decoder' in k}

        to_return.update(state_dict_decoder)

        prefix = kwargs.pop('prefix', '')
        state_dict_mllm = self.mllm.state_dict(*args, prefix=prefix + 'mllm.', **kwargs)
        to_return.update(state_dict_mllm)
        return to_return

    def calc_xyt_token_loss(self, x_mu, y_mu, t_mu, token_predict, data):
        gt_scanpath_x = data.pop('scanpath_x', None) 
        gt_scanpath_y = data.pop('scanpath_y',None) 
        gt_scanpath_t = data.pop('scanpath_t',None) 
        task = data.pop('task',None)

        #如果超出长度，需要截断 
        truncate_len = self.max_predict_lens  
        gt_scanpath_x = gt_scanpath_x[:,:truncate_len]
        gt_scanpath_y = gt_scanpath_y[:,:truncate_len]
        gt_scanpath_t = gt_scanpath_t[:,:truncate_len]

        # 填充到 max_predict_lens 长度
        pad_length = self.max_predict_lens - gt_scanpath_x.shape[1]
        if pad_length > 0:
            gt_scanpath_x = F.pad(gt_scanpath_x, (0, pad_length), mode='constant', value=0)
            gt_scanpath_y = F.pad(gt_scanpath_y, (0, pad_length), mode='constant', value=0)
            gt_scanpath_t = F.pad(gt_scanpath_t, (0, pad_length), mode='constant', value=0)
        
        gt_scanpath_x_flatten = torch.flatten(gt_scanpath_x)
        gt_scanpath_y_flatten = torch.flatten(gt_scanpath_y)
        gt_scanpath_t_flatten = torch.flatten(gt_scanpath_t)

        no_zero_pos_mask = torch.logical_not(gt_scanpath_x_flatten == 0)
        fixation_cnt = no_zero_pos_mask.sum() + 1e-5

        # 计算 loss
        loss_x = (self.loss_fn_xy(x_mu, gt_scanpath_x_flatten) * no_zero_pos_mask).sum() / fixation_cnt
        loss_y = (self.loss_fn_xy(y_mu, gt_scanpath_y_flatten) * no_zero_pos_mask).sum() / fixation_cnt
        loss_t = (self.loss_fn_t(t_mu, gt_scanpath_t_flatten) * no_zero_pos_mask).sum() / fixation_cnt
        
        # 计算 token loss
        token_gt = (gt_scanpath_x == 0).long() 
        
        # loss_fn_token = self.loss_fn_token_dict[task[0]] #只有bs为0的时候在可以，默认用第0个
        # loss_fn_token.weight = loss_fn_token.weight.to(t_mu.device).to(t_mu.dtype)
        # loss_fn_token = self.loss_fn_token
        loss_token = self.loss_fn_token(token_predict.permute(1, 2, 0),
                                       token_gt.flatten(1).long())

        # 计算DTW_loss
        # idx = (token_gt[0] == 1).nonzero()
        # gt_len = idx[0].item() if idx.numel() > 0 else token_gt.shape[1]

        # token_states = torch.argmax(token_predict, dim=-1).squeeze()
        # idx = (token_states == 1).nonzero()
        # pred_len = idx[0].item() if idx.numel() > 0 else token_gt.shape[1]  

        # gt_scanpath_x = gt_scanpath_x.unsqueeze(dim=-1) #[1,7,1]
        # gt_scanpath_y = gt_scanpath_y.unsqueeze(dim=-1)
        # gt_scanpath_xy = torch.cat([gt_scanpath_x, gt_scanpath_y], dim=-1).to(torch.float32) # [1,7,2]

        # pred_scanpath_x = x_mu.unsqueeze(dim=0).unsqueeze(dim=-1) #[1,7,1] #需要确保bs==1
        # pred_scanpath_y = y_mu.unsqueeze(dim=0).unsqueeze(dim=-1) #[1,7,1]
        # pred_scanpath_xy = torch.cat([pred_scanpath_x, pred_scanpath_y], dim=-1).to(torch.float32) # [1,7,2]

        # batch_size = gt_scanpath_x.shape[0]
        # loss_dtw = 0
        # for b in range(batch_size):
        #     # tmp1 = gt_scanpath_xy[b][:gt_len,:]
        #     # tmp2 = pred_scanpath_xy[b][:pred_len,:]
        #     # dtw_distances = torch.cdist(pred_scanpath_xy[b][:pred_len,:], gt_scanpath_xy[b][:gt_len,:])
        #     dtw_distances = torch.cdist(gt_scanpath_xy[b][:gt_len,:],pred_scanpath_xy[b][:gt_len,:])

        #     dtw_path = dtw_distances.min(dim=1)[0].mean()  # 计算最小距离并取平均
        #     loss_dtw += dtw_path
        # loss_dtw /= batch_size
        
        return loss_x, loss_y, loss_t, loss_token
    
    def forward(self, data, data_samples=None, mode='loss'):
        input_ids = data['input_ids']

        output = self.mllm(data, data_samples, mode)
        hidden_states = output.hidden_states[-1]
        del output.hidden_states
        # ref_encode_see = self.tokenizer.decode(input_ids[0])

        # 找到每个样本中第一个 end_token 的索引
        seg_token_mask = input_ids == self.seg_token_idx
        end_token_indices = torch.argmax(seg_token_mask.int(), dim=1) 
        pred_embeddings_vl = hidden_states[:, end_token_indices, :]
        pred_embeddings_vl = pred_embeddings_vl[0] #需要确保bs==1

        vision_start_token_mask = input_ids == self.vision_start_token_idx
        vision_end_token_mask = input_ids == self.vision_end_token_idx
        vision_start_token_indices = torch.argmax(vision_start_token_mask.int(), dim=1)
        vision_end_token_indices = torch.argmax(vision_end_token_mask.int(), dim=1)
        assert vision_start_token_indices[0]==3
        assert vision_end_token_indices[0]==420

        pred_embeddings_vision = torch.stack([
            hidden_states[b, vision_start_token_indices[b]+1:vision_end_token_indices[b], :] for b in range(input_ids.size(0))]).permute(1,0,2)

        outs = self.Scanpath_Decoder(vl_guidance_feats=pred_embeddings_vl, pred_embeddings_vision=pred_embeddings_vision)

        y_mu = self.activation(self.generator_y_mu(outs)).flatten() 
        x_mu = self.activation(self.generator_x_mu(outs)).flatten() 
        t_mu = self.activation(self.generator_t_mu(outs)).flatten() 
        token_predict = self.softmax(self.token_predictor(outs))

        x_mu = torch.clamp(x_mu, min=0, max=512 - 1)  
        y_mu = torch.clamp(y_mu, min=0, max=320 - 1)

        loss_x, loss_y, loss_t, loss_token= self.calc_xyt_token_loss(x_mu, y_mu, t_mu, token_predict, data)

        llm_loss = output.loss

        loss_dict = {
            'loss_xy': loss_x + loss_y,
            't_loss': loss_t,
            'loss_token': loss_token,
            'llm_loss': llm_loss,
        }
        return loss_dict


