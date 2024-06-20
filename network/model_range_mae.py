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
from network.model_utils import init_weights
from einops import rearrange
from einops.layers.torch import Rearrange
from network.decode_head import MyHead
from scipy.spatial.ckdtree import cKDTree as kdtree



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
        self.range_img_size = (64, 2048)

        self.img_mask_ratio, self.range_mask_ratio = 0, 0
        self.range_image_patch = (32, 256)

        self.scale_factor = (2, 8)

        # head & decoder
        self.img_mask_token = nn.Parameter(torch.zeros(1, 1, config['model_params']['mae_parameters']['embed_dim']))
        self.img_decoder_pos_embed = nn.Parameter(torch.zeros(1, self.img_patch[0]*self.img_patch[1] + 1, config['model_params']['mae_parameters']['embed_dim']), requires_grad=False)  # fixed sin-cos embedding
        self.range_mask_token = nn.Parameter(torch.zeros(1, 1, config['model_params']['mae_parameters']['embed_dim']))
        self.range_decoder_pos_embed = nn.Parameter(torch.zeros(1, self.range_image_patch[0]*self.range_image_patch[1] + 1, config['model_params']['mae_parameters']['embed_dim']), requires_grad=False)  # fixed sin-cos embedding

        # self.head = ATMHead(config=config['model_params'])

        self.head = MyHead(config=config['model_params'])

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

    def forward_decode_seg_range(self, range_latent_full, range_ids_restore_full, img_token_2_range, range_skip, batch_dict):
        range_latent_full[:, 1:, :] += img_token_2_range[:, 1:, :]

        h, w = self.range_image_patch
        range_feat = range_latent_full[:, 1:, :]

        px, py = batch_dict['laser_x'], batch_dict['laser_y']
        points_xyz = batch_dict['points']
        knns = batch_dict['knns']
        num_points = batch_dict['num_points']

        px = px[:,-1]
        py = py[:, -1]
        points_xyz = points_xyz[:, 1:]
        h, w = self.range_img_size
        px = 2.0 * ((px / w) - 0.5)
        py = 2.0 * ((py / h) - 0.5)
        mask3d = self.head(range_feat, range_skip, px, py, points_xyz, knns, num_points)
        return mask3d.squeeze()
    
    def forward(self, batch_dict):
        laser_range_in = batch_dict['laser_range_in'].permute(0,3,1,2).contiguous()
        laser_x = batch_dict['laser_x']
        laser_y = batch_dict['laser_y']
        raw_images = batch_dict['img']
        # B = batch_dict['batch_size']
        Batch_size, _, H_raw, W_raw = raw_images.size() # (256, 1024)

        images = torch.nn.functional.interpolate(raw_images, size=self.img_size, mode='bilinear')
        img_latent_full, img_mask_full, img_ids_restore_full, img_skip = self.mae.image_encoder.forward_encoder(images, self.img_mask_ratio)
        range_latent_full, range_mask_full, range_ids_restore_full, range_skip = self.mae.range_encoder.forward_encoder(laser_range_in, self.range_mask_ratio)

        img_latent_full_cls, img_mask_full_tokens_cls_unpatch, range_latent_full_cls, range_mask_full_tokens_cls_unpatch = self.forward_patchfy_unpatchfy_img_range(img_latent_full, img_mask_full, img_ids_restore_full,
                                                                                                                        range_latent_full, range_mask_full, range_ids_restore_full)
        
        

        D = img_mask_full_tokens_cls_unpatch.shape[-1]
        h_img, w_img = self.mae.range_img_size[0] // self.mae.range_patch_size[0], self.mae.range_img_size[1] // self.mae.range_patch_size[1]
        img_token_2_range = Variable(torch.zeros((Batch_size, h_img, w_img, D), dtype=range_latent_full.dtype, device=range_latent_full.device))
        img_token_2_range[:, :, 48:80, :] = torch.nn.functional.interpolate(img_mask_full_tokens_cls_unpatch.permute(0, 3, 1, 2).contiguous(), size=(h_img, 32), mode='bilinear').permute(0, 2, 3, 1).contiguous().detach()
        img_token_2_range.requires_grad=True

        img_token_2_range = img_token_2_range.reshape(Batch_size, -1, D)
        img_token_2_range = torch.cat([img_latent_full_cls, img_token_2_range], 1)

        range_pred = self.forward_decode_seg_range(range_latent_full, range_ids_restore_full, img_token_2_range, range_skip, batch_dict)
        batch_dict['loss'] = 0.
        batch_dict['logits'] = range_pred
        batch_dict = self.criterion(batch_dict)


        return batch_dict
