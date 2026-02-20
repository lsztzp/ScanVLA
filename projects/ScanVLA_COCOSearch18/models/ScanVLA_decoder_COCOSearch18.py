from typing import Optional
import copy

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from timm.models.layers import Mlp
from xtuner.registry import BUILDER

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

def modulate(x, shift, scale, only_first=False):
    if only_first:
        x_first, x_rest = x[:, :1], x[:, 1:]
        x = torch.cat([x_first * (1 + scale.unsqueeze(0)) + shift.unsqueeze(0), x_rest], dim=1)
    else:
        x = x * (1 + scale.unsqueeze(0)) + shift.unsqueeze(0)

    return x

class TransformerDecoderWrapper(nn.Module):
    def __init__(self, activation,                 
                hidden_dim=256, 
                nhead=8, 
                dim_feedforward=1024, 
                dropout_attn = 0.1,
                dropout_mlp = 0.15,
                num_decoder_layers=6, 
                max_len=4, 
                torch_dtype=torch.bfloat16, 
                args=None):
        super().__init__()
        self.hidden_dim = hidden_dim

        decoder_layer = TransformerDecoderLayer(d_model = self.hidden_dim, nhead = nhead, dim_feedforward = dim_feedforward,
                                                dropout_attn = dropout_attn, dropout_mlp = dropout_mlp)
        
        final_layer = FinalLayer(2048, hidden_dim) # Diffusion-Planner的Finallayer

        # decoder_norm = nn.LayerNorm(self.hidden_dim)
        decoder_norm = None
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=False, final_layer=final_layer)

        self.query_embed = nn.Embedding(max_len, self.hidden_dim)
        self.xy_embed = nn.Embedding(416, self.hidden_dim)

        self.query_len = max_len

        self._reset_parameters()

        self.dtype = torch.bfloat16
        self.to(self.dtype)
        
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_seg_embed(self, tensor, seg: Optional[Tensor]):
        return tensor if seg is None else tensor + seg
        
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else pos + tensor

    def forward(self, 
                vl_guidance_feats, 
                pred_embeddings_vision,
                ):  
          
        bs = pred_embeddings_vision.size(1)  # L，B, D

        # 查询信息
        query = self.with_pos_embed(torch.zeros(self.query_len, bs, self.hidden_dim, device=self.query_embed.weight.device, dtype=vl_guidance_feats.dtype), 
                                  pos=self.query_embed.weight.unsqueeze(1))

        # 查询信息和历史注视点编码和vl_guidance_feats融合
        querypos_embed = self.query_embed.weight.unsqueeze(1)
        visionpos_embed = None

        output = self.decoder(query,
                              vl_guidance_feats=vl_guidance_feats,
                              pred_embeddings_vision=pred_embeddings_vision,
                              querypos_embed = querypos_embed,
                                visionpos_embed = visionpos_embed) #1
        return output


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False, final_layer=None):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        
        self.final_layer = final_layer

    def forward(self, 
                query,
                vl_guidance_feats,
                pred_embeddings_vision,
                querypos_embed,
                visionpos_embed):
        output = query

        intermediate = []

        for idx, layer in enumerate(self.layers):
            output = layer(output, 
                           vl_guidance_feats,
                           pred_embeddings_vision,
                           querypos_embed,
                           visionpos_embed
                           )
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.final_layer:
            output = self.final_layer(output, vl_guidance_feats)

        # 不强制要求norm化输出
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output
    

class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model=256, nhead=8, dim_feedforward=1024, dropout_attn=0.1, dropout_mlp=0.15, torch_dtype=torch.bfloat16):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout_attn)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout_attn)

        self.norm_vl = nn.LayerNorm(2048, eps=1e-5) #对vl_feature进行LN
        self.norm_vision = nn.LayerNorm(2048, eps=1e-5) #对vision feature进行LN
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout_attn)
        self.dropout2 = nn.Dropout(dropout_attn)

        self.Linear_vl = nn.Sequential(
            nn.SiLU(),
            nn.Linear(2048, d_model, bias=True)
        )

        # adaLN调制模块
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model, bias=True)
        )

        # 将视觉token转换
        self.Linear_vision = nn.Sequential(
            nn.SiLU(),
            nn.Linear(2048, d_model, bias=True)
        )

        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp1 = Mlp(in_features=d_model, hidden_features=dim_feedforward, act_layer=approx_gelu, drop=dropout_mlp)
        self.mlp2 = Mlp(in_features=d_model, hidden_features=dim_feedforward, act_layer=approx_gelu, drop=dropout_mlp)

        self.dtype = torch.bfloat16
        self.to(self.dtype)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else pos + tensor

    def forward_AdaLN(self, tgt, 
                            vl_guidance_feats,
                            pred_embeddings_vision, 
                            querypos_embed,
                            visionpos_embed):
        
        vl_guidance_feats =self.norm_vl(vl_guidance_feats)
        vl_guidance_feats = self.Linear_vl(vl_guidance_feats) #转移维度
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(vl_guidance_feats).chunk(6, dim=1)
        modulated_tgt = modulate(self.norm1(tgt), shift_msa, scale_msa)
        q = k = v = self.with_pos_embed(modulated_tgt, querypos_embed)

        tgt2 = self.self_attn(q, k, value=v, attn_mask=None,
                              key_padding_mask=None)[0]
        tgt = tgt + self.dropout1(gate_msa.unsqueeze(0) * tgt2)

        modulated_tgt = modulate(self.norm2(tgt), shift_mlp, scale_mlp)
        tgt = tgt + gate_mlp.unsqueeze(0) * self.mlp1(modulated_tgt)

        # 和视觉信息进行交叉注意力
        pred_embeddings_vision = self.norm_vision(pred_embeddings_vision)
        pred_embeddings_vision = self.Linear_vision(pred_embeddings_vision)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(self.norm3(tgt), querypos_embed),
                                   key=self.with_pos_embed(pred_embeddings_vision, visionpos_embed), #图像添加位置编码
                                   value=pred_embeddings_vision, attn_mask=None,
                                   key_padding_mask=None)[0]
        tgt = tgt + self.dropout2(tgt2)

        # FFN
        tgt2 = self.mlp2(self.norm4(tgt))
        tgt = tgt + tgt2

        return tgt

    def forward(self, 
                query,
                vl_guidance_feats,
                pred_embeddings_vision,
                querypos_embed,
                visionpos_embed):
        
        return self.forward_AdaLN(query, 
                                    vl_guidance_feats,
                                    pred_embeddings_vision, 
                                    querypos_embed,
                                      visionpos_embed)

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, input_size, hidden_size, torch_dtype=torch.bfloat16):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size)
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 4, bias=True),
            nn.GELU(approximate="tanh"),
            nn.LayerNorm(hidden_size * 4),
            nn.Linear(hidden_size * 4, hidden_size, bias=True)
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(input_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, y):
        L, B, _ = x.shape
        
        shift, scale = self.adaLN_modulation(y).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.proj(x)
        return x


