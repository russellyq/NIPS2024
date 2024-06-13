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
from network.model_utils import get_grid_size_1d, get_grid_size_2d, init_weights
from einops import rearrange
from einops.layers.torch import Rearrange
from network.kpconv.blocks import KPConv

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


class MyHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.use_stages = config['use_stages']
        self.nhead = config['decode_seg_nhead']
        self.dim = config['decode_seg_dim']
        self.num_layers = config['decode_seg_nlayers']
        dropout_rate = 0.2
        self.range_patch = (2, 8)
        self.range_image_size = (64, 1024)
        self.patch_num = (32, 128)
        skip_filters = config['skip_filters']

        self.up_conv_block = UpConvBlock(
            self.dim, self.dim,
            dropout_rate=dropout_rate,
            scale_factor=self.range_patch,
            drop_out=False,
            skip_filters=skip_filters)
        
        self.pred = KPClassifier(in_channels=self.dim,
                                    out_channels=self.dim,
                                    num_classes=config['num_classes'])

    
    def forward(self, x, skip, px, py, pxyz, pknn, num_points):
        
        GS_H, GS_W = self.patch_num
        x = rearrange(x, 'b (h w) c -> b c h w', h=GS_H, w=GS_W) # B, d_model, image_size[0]/patch_stride[0], image_size[1]/patch_stride[1]
        feats = self.up_conv_block(x, skip)
        
        masks3d = self.pred(x, px, py, pxyz, pknn, num_points)

        return masks3d

class UpConvBlock(nn.Module):
    def __init__(
        self,
        in_filters,
        out_filters,
        dropout_rate,
        scale_factor=(2, 8),
        drop_out=False,
        skip_filters=0):
        super(UpConvBlock, self).__init__()

        self.in_filters = in_filters
        self.out_filters = out_filters
        self.skip_filters = skip_filters

        # scale_factor has to be a tuple or a list with two elements
        if isinstance(scale_factor, int):
            scale_factor = (scale_factor, scale_factor)
        assert isinstance(scale_factor, (list, tuple))
        assert len(scale_factor) == 2
        self.scale_factor = scale_factor

        if self.scale_factor[0] != self.scale_factor[1]:
            upsample_layers = [
                nn.Conv2d(in_filters, out_filters * self.scale_factor[0] * self.scale_factor[1], kernel_size=(1, 1)),
                Rearrange('b (c s0 s1) h w -> b c (h s0) (w s1)', s0=self.scale_factor[0], s1=self.scale_factor[1]),]
        else:
            upsample_layers = [
                nn.Conv2d(in_filters, out_filters * self.scale_factor[0] * self.scale_factor[1], kernel_size=(1, 1)),
                nn.PixelShuffle(self.scale_factor[0]),]

        if drop_out:
            upsample_layers.append(nn.Dropout2d(p=dropout_rate))
        self.conv_upsample = nn.Sequential(*upsample_layers)

        self.conv1 = nn.Sequential(
            nn.Conv2d(out_filters + skip_filters, out_filters, (3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_filters)
        )
        num_filters = out_filters
        output_layers = [
            nn.Conv2d(num_filters, out_filters, kernel_size=(1, 1)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_filters),
        ]
        if drop_out:
            output_layers.append(nn.Dropout2d(p=dropout_rate))
        self.conv_output = nn.Sequential(*output_layers)

    def forward(self, x, skip=None):
        x_up = self.conv_upsample(x) # increase spatial size by a scale factor. B, 2*base_channels, image_size[0], image_size[1]

        if self.skip_filters > 0:
            assert skip is not None
            assert skip.shape[1] == self.skip_filters
            x_up = torch.cat((x_up, skip), dim=1)

        x_up_out = self.conv_output(self.conv1(x_up))
        return x_up_out

class KPClassifier(nn.Module):
    # Adapted from D. Kochanov et al.
    # https://github.com/DeyvidKochanov-TomTom/kprnet
    def __init__(self, in_channels=256, out_channels=256, num_classes=17, dummy=False):
        super(KPClassifier, self).__init__()
        self.kpconv = KPConv(
            kernel_size=15,
            p_dim=3,
            in_channels=in_channels,
            out_channels=out_channels,
            KP_extent=1.2,
            radius=0.60,
        )
        self.dummy = dummy
        if self.dummy:
            del self.kpconv
            self.kpconv = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.head = nn.Conv2d(out_channels, num_classes, 1)

    def forward(self, x, px, py, pxyz, pknn, num_points):
        assert px.shape[0] == py.shape[0]
        assert px.shape[0] == pxyz.shape[0]
        assert px.shape[0] == pknn.shape[0]
        assert px.shape[0] == num_points.sum().item()
        res = []
        offset = 0
        batch_size = x.shape[0]
        for i in range(batch_size):
            len = num_points[i]
            px_i = px[offset:(offset+len)].unsqueeze(0).unsqueeze(1).contiguous()
            py_i = py[offset:(offset+len)].unsqueeze(0).unsqueeze(1).contiguous()
            points = pxyz[offset:(offset+len)].contiguous()
            pknn_i = pknn[offset:(offset+len)].contiguous()
            resampled = F.grid_sample(
                x[i].unsqueeze(0), torch.stack([px_i, py_i], dim=3),
                align_corners=False, padding_mode='border')
            feats = resampled.squeeze().t()

            if feats.shape[0] != points.shape[0]:
                print(f'feats.shape={feats.shape} vs points.shape={points.shape}')
            assert feats.shape[0] == points.shape[0]
            if self.dummy:
                feats = self.kpconv(feats)
            else:
                feats = self.kpconv(points, points, pknn_i, feats)
            res.append(feats)
            offset += len

        assert offset == px.shape[0]
        res = torch.cat(res, axis=0).unsqueeze(2).unsqueeze(3)
        res = self.relu(self.bn(res))
        return self.head(res)


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
