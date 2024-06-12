import torch
import torch_scatter
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from network.basic_block import Lovasz_loss
from network.baseline import get_model as SPVCNN
from network.base_model import LightningBaseModel
from network.basic_block import ResNetFCN
from network.mae import MAE
from torch.autograd import Variable
from network.baseline import criterion
from torch import Tensor
from timm.models.vision_transformer import Block, Mlp
from timm.models.layers import trunc_normal_
from typing import Optional


def trunc_normal_init(module: nn.Module,
                      mean: float = 0,
                      std: float = 1,
                      a: float = -2,
                      b: float = 2,
                      bias: float = 0) -> None:
    if hasattr(module, 'weight') and module.weight is not None:
        trunc_normal_(module.weight, mean, std, a, b)  # type: ignore
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)  # type: ignore

def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

class get_model(LightningBaseModel):
    def __init__(self, config):
        super(get_model, self).__init__(config)
        self.save_hyperparameters()
        self.baseline_only = config.baseline_only
        self.num_classes = config.model_params.num_classes
        self.hiden_size = config.model_params.hiden_size
        self.lambda_seg2d = config.train_params.lambda_seg2d
        self.lambda_xm = config.train_params.lambda_xm
        self.scale_list = config.model_params.scale_list
        self.num_scales = len(self.scale_list)
        self.config = config

        self.mae = MAE(config['model_params']['mae_parameters'])
        self._init_mae_(config.model_params.pretrain)
        self.img_size = (256, 1024)
        self.img_patch = (16, 64)
        self.range_img_size = (64, 1024)

        self.img_mask_ratio, self.range_mask_ratio = 0, 0
        self.range_image_patch = (32, 128)

        self.scale_factor = (2, 8)
        # head & decoder
        self.img_mask_token = nn.Parameter(torch.zeros(1, 1, config['model_params']['mae_parameters']['embed_dim']))
        self.img_decoder_pos_embed = nn.Parameter(torch.zeros(1, self.img_patch[0]*self.img_patch[1] + 1, config['model_params']['mae_parameters']['embed_dim']), requires_grad=False)  # fixed sin-cos embedding
        self.range_mask_token = nn.Parameter(torch.zeros(1, 1, config['model_params']['mae_parameters']['embed_dim']))
        self.range_decoder_pos_embed = nn.Parameter(torch.zeros(1, self.range_image_patch[0]*self.range_image_patch[1] + 1, config['model_params']['mae_parameters']['embed_dim']), requires_grad=False)  # fixed sin-cos embedding

        self.head = ATMHead(config=config['model_params'])

        self.criterion = criterion(config)
        
    def _init_mae_(self, pretrain=False):
        if pretrain:
            self.mae.load_params_from_file('/home/yanqiao/OpenPCDet/output/semantic_kitti_models/MAE/checkpoints/checkpoint_epoch_100.pth', None)
            print('Loaded MAE from pre-trained weights')

            # for param in self.mae.image_encoder.parameters():
            #     param.requires_grad = False
            # for param in self.mae.range_encoder.decoder_blocks.parameters():
            #     param.requires_grad = False
            # for param in self.mae.parameters():
            #     param.requires_grad = False
        else:
            print('vanilla training !')


    def forward_loss(self, pred, batch_dict):
        laser_range_in = batch_dict['laser_range_in'].permute(0,3,1,2).contiguous()
        laser_x = batch_dict['laser_x']
        laser_y = batch_dict['laser_y']
        pred_all = []
        for batch_idx in range(batch_dict['batch_size']):
            x_batch = laser_x[torch.where(laser_x[:, 0] == batch_idx)][:, -1]
            y_batch = laser_y[torch.where(laser_y[:, 0] == batch_idx)][:, -1]
            pre_batch = pred[batch_idx, :, y_batch.long(), x_batch.long()].permute(1, 0).contiguous()
            pred_all.append(pre_batch)
        pred_all = torch.cat(pred_all, dim=0)
        labels = batch_dict['point_labels']
        loss_main_ce, loss_main_lovasz, loss_main = self.criterion(pred_all, labels)
        return loss_main_ce, loss_main_lovasz, loss_main

    def forward_patchfy_unpatchfy_img_range(self, img_latent_full, img_mask_full, img_ids_restore_full,
                                            range_latent_full, range_mask_full, range_ids_restore_full):
        B = img_latent_full.shape[0]
        
        # append mask tokens to sequence
        img_mask_full_tokens = self.img_mask_token.repeat(img_latent_full.shape[0], img_ids_restore_full.shape[1] + 1 - img_latent_full.shape[1], 1) # (B, L2+1-L1=1024-256=768, D)
        x_ = torch.cat([img_latent_full[:, 1:, :], img_mask_full_tokens], dim=1)  # no cls token # (B, L2, D)
        x_ = torch.gather(x_, dim=1, index=img_ids_restore_full.unsqueeze(-1).repeat(1, 1, img_latent_full.shape[2]))  # unshuffle # (B, L2, D)
        
        img_latent_full = torch.cat([img_latent_full[:, :1, :], x_], dim=1)  # append cls token # (B, L2+1, D)

        # add pos embed
        img_latent_full = img_latent_full + self.img_decoder_pos_embed # (B, L2+1, D)
    
        # append mask tokens to sequence
        range_mask_full_tokens = self.range_mask_token.repeat(range_latent_full.shape[0], range_ids_restore_full.shape[1] + 1 - range_latent_full.shape[1], 1) # (B, L2+1-L1=1024-256=768, D)
        x_ = torch.cat([range_latent_full[:, 1:, :], range_mask_full_tokens], dim=1)  # no cls token # (B, L2, D)
        x_ = torch.gather(x_, dim=1, index=range_ids_restore_full.unsqueeze(-1).repeat(1, 1, range_latent_full.shape[2]))  # unshuffle # (B, L2, D)
        
        range_latent_full = torch.cat([range_latent_full[:, :1, :], x_], dim=1)  # append cls token # (B, L2+1, D)

        # add pos embed
        range_latent_full = range_latent_full + self.range_decoder_pos_embed # (B, L2+1, D)
        
        h_img, w_img = 16, 64 # (256, 1024) / (16, 16) = (64, 1024) / (4, 16)
        img_mask_full_tokens_cls_unpatch = img_latent_full[:, 1:, :].reshape(B, h_img, w_img, img_latent_full.shape[-1])
        
        h_img, w_img = self.range_img_size[0] // self.scale_factor[0], self.range_img_size[1] // self.scale_factor[1]
        range_mask_full_tokens_cls_unpatch = range_latent_full[:, 1:, :].reshape(B, h_img, w_img, range_latent_full.shape[-1])

        return img_latent_full[:,:1, :], img_mask_full_tokens_cls_unpatch, range_latent_full[:, :1, :], range_mask_full_tokens_cls_unpatch

    def forward_decode_seg_range(self, range_latent_full, range_ids_restore_full, img_token_2_range, batch_dict):
        range_latent_full[:, 1:, :] += img_token_2_range[:, 1:, :]
        in_put = []
        in_put.append(range_latent_full)
        out_dict = self.head(tuple(in_put))
        return out_dict
    
    def forward(self, batch_dict):
        laser_range_in = batch_dict['laser_range_in'].permute(0,3,1,2).contiguous()
        laser_x = batch_dict['laser_x']
        laser_y = batch_dict['laser_y']
        raw_images = batch_dict['img']
        # B = batch_dict['batch_size']
        Batch_size, _, H_raw, W_raw = raw_images.size() # (256, 1024)

        images = torch.nn.functional.interpolate(raw_images, size=self.img_size, mode='bilinear')
        img_latent_full, img_mask_full, img_ids_restore_full = self.mae.image_encoder.forward_encoder(images, self.img_mask_ratio)
        range_latent_full, range_mask_full, range_ids_restore_full = self.mae.range_encoder.forward_encoder(laser_range_in, self.range_mask_ratio)
        
        img_latent_full_cls, img_mask_full_tokens_cls_unpatch, range_latent_full_cls, range_mask_full_tokens_cls_unpatch = self.forward_patchfy_unpatchfy_img_range(img_latent_full, img_mask_full, img_ids_restore_full,
                                                                                                                        range_latent_full, range_mask_full, range_ids_restore_full)
        
        

        D = img_mask_full_tokens_cls_unpatch.shape[-1]
        h_img, w_img = self.mae.range_img_size[0] // self.mae.range_patch_size[0], self.mae.range_img_size[1] // self.mae.range_patch_size[1]
        img_token_2_range = Variable(torch.zeros((Batch_size, h_img, w_img, D), dtype=range_latent_full.dtype, device=range_latent_full.device))
        img_token_2_range[:, :, 48:80, :] = torch.nn.functional.interpolate(img_mask_full_tokens_cls_unpatch.permute(0, 3, 1, 2).contiguous(), size=(h_img, 32), mode='bilinear').permute(0, 2, 3, 1).contiguous().detach()
        img_token_2_range.requires_grad=True

        img_token_2_range = img_token_2_range.reshape(Batch_size, -1, D)
        img_token_2_range = torch.cat([img_latent_full_cls, img_token_2_range], 1)

        range_pred = self.forward_decode_seg_range(range_latent_full, range_ids_restore_full, img_token_2_range, batch_dict)
        
        batch_dict['loss'] = 0.
        laser_range_in = batch_dict['laser_range_in'].permute(0,3,1,2).contiguous()
        laser_x = batch_dict['laser_x']
        laser_y = batch_dict['laser_y']
        pred_all = []
        for batch_idx in range(batch_dict['batch_size']):
            x_batch = laser_x[torch.where(laser_x[:, 0] == batch_idx)][:, -1]
            y_batch = laser_y[torch.where(laser_y[:, 0] == batch_idx)][:, -1]
            pre_batch = range_pred[batch_idx, :, y_batch.long(), x_batch.long()].permute(1, 0).contiguous()
            pred_all.append(pre_batch)
        pred_all = torch.cat(pred_all, dim=0)

        batch_dict['logits'] = pred_all
        batch_dict = self.criterion(batch_dict)


        return batch_dict

class ATMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.use_stages = config['use_stages']
        self.nhead = config['decode_seg_nhead']
        self.dim = config['decode_seg_dim'] // 2
        self.num_layers = config['decode_seg_nlayers']

        self.range_patch = (2, 8)
        self.range_image_size = (64, 1024)
        self.patch_num = (32, 128)

        input_proj = []
        proj_norm = []
        atm_decoders = []

        for i in range(self.use_stages):
            # proj
            proj = nn.Linear(config['decode_seg_dim'], self.dim)
            trunc_normal_(proj.weight, std=.02)
            self.add_module("input_proj_{}".format(i + 1), proj)
            input_proj.append(proj)

            # norm
            norm = nn.LayerNorm(self.dim)
            self.add_module("proj_norm_{}".format(i + 1), norm)
            proj_norm.append(norm)

            # decoder layer
            decoder_layer = TPN_DecoderLayer(d_model=self.dim, nhead=self.nhead, dim_feedforward=self.dim * 4)
            decoder = TPN_Decoder(decoder_layer, self.num_layers)
            self.add_module("decoder_{}".format(i + 1), decoder)
            atm_decoders.append(decoder)
        
        self.input_proj = input_proj
        self.proj_norm = proj_norm
        self.decoder = atm_decoders
        self.q = nn.Embedding(config['num_classes'], self.dim)

        self.class_embed = nn.Linear(self.dim, config['num_classes'])

        self.init_weights()
        print('successfully initilization of Seg Head !')
    
    def init_weights(self):
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=.02, bias=0)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, val=1.0, bias=0.0)
    
    def forward(self, inputs):
        x = []
        for stage_ in inputs[:self.use_stages]:
            x.append(stage_)
        x.reverse()
        bs = x[0].size()[0]

        laterals = []
        attns = []
        maps_size = []
        qs = []
        q = self.q.weight.repeat(bs, 1, 1).transpose(0, 1)

        for idx, (x_, proj_, norm_, decoder_) in enumerate(zip(x, self.input_proj, self.proj_norm, self.decoder)):
            lateral = norm_(proj_(x_))
            
            laterals.append(lateral)

            q, attn = decoder_(q, lateral.transpose(0, 1))
            attn = attn.transpose(-1, -2)

            attn = self.d3_to_d4(attn)
            maps_size.append(attn.size()[-2:])
            qs.append(q.transpose(0, 1))
            attns.append(attn)
        qs = torch.stack(qs, dim=0)
        outputs_class = self.class_embed(qs)
        out = {"pred_logits": outputs_class[-1]}

        outputs_seg_masks = []
        size = maps_size[-1]
        for i_attn, attn in enumerate(attns):
            if i_attn == 0:
                outputs_seg_masks.append(F.interpolate(attn, size=size, mode='bilinear', align_corners=False))
            else:
                outputs_seg_masks.append(outputs_seg_masks[i_attn - 1] +
                                         F.interpolate(attn, size=size, mode='bilinear', align_corners=False))

        out["pred_masks"] = F.interpolate(outputs_seg_masks[-1],
                                          size=self.range_image_size,
                                          mode='bilinear', align_corners=False)

        pred_logits = self.semantic_inference(out["pred_logits"], out["pred_masks"])

        return pred_logits
      
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_masks": b}
            for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
        ]
    
    def semantic_inference(self, mask_cls, mask_pred):
        # mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        # mask_pred = mask_pred.sigmoid()
        # semseg = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred)
        
        mask_cls = mask_cls
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred)
        return semseg


    def d3_to_d4(self, t,):
        n, hw, c = t.size()
        if hw % 2 != 0:
            t = t[:, 1:, :]
        h, w = self.patch_num
        # print('h,w,hw:', h,w,hw)
        # assert h * w == hw
        return t.transpose(1, 2).reshape(n, c, h, w)
    

class TPN_Decoder(TransformerDecoder):
    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None):
        output = tgt
        # attns = []
        for mod in self.layers:
            output, attn = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)
            # attns.append(attn)

        if self.norm is not None:
            output = self.norm(output)

        return output, attn

class TPN_DecoderLayer(TransformerDecoderLayer):
    def __init__(self, **kwargs):
        super(TPN_DecoderLayer, self).__init__(**kwargs)
        del self.multihead_attn
        self.multihead_attn = Attention(
            kwargs['d_model'], num_heads=kwargs['nhead'], qkv_bias=True, attn_drop=0.1)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, attn2 = self.multihead_attn(
            tgt.transpose(0, 1), memory.transpose(0, 1), memory.transpose(0, 1))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, attn2

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, xq, xk, xv):
        B, Nq, C = xq.size()
        Nk = xk.size()[1]
        Nv = xv.size()[1]

        q = self.q(xq).reshape(B, Nq, self.num_heads,
                                      C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(xk).reshape(B, Nk, self.num_heads,
                                      C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(xv).reshape(B, Nv, self.num_heads,
                                      C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_save = attn.clone()
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x.transpose(0, 1), attn_save.sum(dim=1) / self.num_heads
