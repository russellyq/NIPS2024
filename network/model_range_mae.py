import torch
import torch_scatter
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from network.basic_block import Lovasz_loss
from network.baseline import get_model as SPVCNN
from network.base_model import LightningBaseModel
from network.basic_block import ResNetFCN
from network.mae import MAE
from torch.autograd import Variable
from network.baseline import criterion


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

        self.mae = MAE()
        self._init_mae_(config.model_params.pretrain)
        self.img_size = (256, 1024)
        self.range_img_size = (64, 1024)

        self.img_mask_ratio, self.range_mask_ratio = 0, 0
        self.range_image_patch = (8, 128)
        decode_layer = [512, 256, 128, 64]
        self.decoder_pred = nn.ModuleList()
        for i in range(len(decode_layer) - 1):
            self.decoder_pred.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    nn.Conv2d(decode_layer[i], decode_layer[i+1], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(decode_layer[i+1]),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(decode_layer[i+1], decode_layer[i+1], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(decode_layer[i+1]),
                    nn.ReLU(inplace=True)
                )
            )
        self.decoder_pred.append(
            nn.Sequential(
                nn.Conv2d(64, config['model_params']['num_classes'], kernel_size=1)
            )
        )
        self.criterion = criterion(config)
        
    
    def _init_mae_(self, pretrain=False):
        if pretrain:
            self.mae.load_params_from_file('/home/yanqiao/OpenPCDet/output/semantic_kitti_models/MAE/checkpoints/checkpoint_epoch_100.pth', None)
            print('Loaded MAE from pre-trained weights')
        else:
            print('vanilla training !')

    
    
    def forward_decode(self, x, ids_restore, x_mm, \
                        decoder_embed, \
                        mask_token, \
                        decoder_pos_embed):
        # embed tokens
        x = decoder_embed(x) # (B, L1, D=512)
        B, _, D = x.size()
        # append mask tokens to sequence
        mask_tokens = mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1) # (B, L2+1-L1=1024-256=768, D)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token # (B, L2, D)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle # (B, L2, D)
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token # (B, L2+1, D)
        # add pos embed
        x = x + decoder_pos_embed # (B, L2+1, D)
        x = x + x_mm
        x = x[:, 1:, :].reshape(B, self.range_image_patch[0], self.range_image_patch[1], D).permute(0, 3, 1, 2).contiguous()
        # apply Transformer blocks
        for blk in self.decoder_pred:
            x = blk(x)
        return x
    
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
        # from IPython import embed; embed()
        loss_main_ce, loss_main_lovasz, loss_main = self.criterion(pred_all, labels)
        return loss_main_ce, loss_main_lovasz, loss_main
  
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

        img_latent_full_cls, img_mask_full_tokens_cls_unpatch, range_latent_full_cls, range_mask_full_tokens_cls_unpatch = self.mae.forward_patchfy_unpatchfy_img_range(img_latent_full, img_mask_full, img_ids_restore_full,
                                                                                                                        range_latent_full, range_mask_full, range_ids_restore_full)
        
        D = img_mask_full_tokens_cls_unpatch.shape[-1]
        h_img, w_img = self.mae.range_img_size[0] // self.mae.range_patch_size[0], self.mae.range_img_size[1] // self.mae.range_patch_size[1]
        img_token_2_range = Variable(torch.zeros((Batch_size, h_img, w_img, D), dtype=range_latent_full.dtype, device=range_latent_full.device))
        img_token_2_range[:, :, 48:80, :] = torch.nn.functional.interpolate(img_mask_full_tokens_cls_unpatch.permute(0, 3, 1, 2).contiguous(), size=(h_img, 32), mode='bilinear').permute(0, 2, 3, 1).contiguous().detach()
        img_token_2_range.requires_grad=True

        img_token_2_range = img_token_2_range.reshape(Batch_size, -1, D)
        img_token_2_range = torch.cat([img_latent_full_cls, img_token_2_range], 1)

        

        range_pred = self.forward_decode(range_latent_full, range_ids_restore_full, img_token_2_range,
                                        self.mae.range_encoder.decoder_embed,
                                        self.mae.range_encoder.mask_token,
                                        self.mae.range_encoder.decoder_pos_embed,
                                        )
        
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