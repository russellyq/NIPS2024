import torch
import torch_scatter
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

from network.basic_block import Lovasz_loss
from network.baseline import get_model as SPVCNN
from network.base_model import LightningBaseModel
from network.basic_block import ResNetFCN
from network.mae_spvcnn import SPVCNN_MAE
from torch.autograd import Variable
from network.baseline import criterion

from timm.models.vision_transformer import Block, Mlp

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

        self.img_size = (256, 1024)
        self.img_mask_ratio = 0

        self.spvcnn_mae = SPVCNN_MAE(config['model_params']['mae_parameters'], sample_points=False)
        
        self.criterion = criterion(config)

        self.multihead_3d_classifier = nn.ModuleList()
        for i in range(self.num_scales):
            self.multihead_3d_classifier.append(
                nn.Sequential(
                    nn.Linear(self.hiden_size, 128),
                    nn.ReLU(True),
                    nn.Linear(128, self.num_classes))
            )
        
    def _init_mae_(self, pretrain=False):
        if pretrain:
            self.mae.load_params_from_file('/home/yanqiao/OpenPCDet/output/semantic_kitti_models/MAE/checkpoints/checkpoint_epoch_100.pth', None)
            print('Loaded MAE from pre-trained weights')
            for param in self.mae.parameters():
                param.requires_grad = True
            for param in self.mae.image_encoder.parameters():
                param.requires_grad = False
        else:
            print('vanilla training !')

    def forward(self, batch_dict):
        batch_dict['spconv_points'] = batch_dict['points']
        raw_images = batch_dict['img']
        # B = batch_dict['batch_size']
        Batch_size, _, H_raw, W_raw = raw_images.size() # (256, 1024)
        images = torch.nn.functional.interpolate(raw_images, size=self.img_size, mode='bilinear')

        # color image encoding
        img_latent, img_mask, img_ids_restore = self.spvcnn_mae.image_encoder.forward_encoder(images, self.img_mask_ratio)
        # img_latent_full, img_mask_full, img_ids_restore_full = self.image_encoder.forward_encoder(images, 0)
        img_latent_full = self.forward_decoder_img(img_latent, img_ids_restore)
        img_latent_full = self.img_conv(img_latent_full.reshape(Batch_size, self.img_size[0]//self.scale_factor[0], self.img_size[1]//self.scale_factor[1], -1).permute(0, 3, 1, 2).contiguous())
        # img_latent_full: (B, C, H, W)

        # spvcnn:
        batch_dict = self.spvcnn_mae.pc_encoder(batch_dict)

        for idx in range(4):
            last_scale = self.scale_list[idx - 1] if idx > 0 else 1
            points_img = batch_dict['points_img']
            point2img_index = batch_dict['point2img_index'] # list # (N_pc2img, N_pc2img, ..., ... )
            batch_idx = batch_dict['batch_idx']
            pts_feat_f = batch_dict['spconv_points_layer_{}'.format(idx)]['pts_feat_f']
            pts_feat = batch_dict['spconv_points_layer_{}'.format(idx)]['pts_feat']
            coors_inv = batch_dict['spconv_points_scale_{}'.format(last_scale)]['coors_inv']

            # 3D prediction
            pts_pred_full = self.multihead_3d_classifier[idx](pts_feat)
            
            # correspondence
            pts_label_full = self.voxelize_labels(batch_dict['labels'], batch_dict['spconv_points_layer_{}'.format(idx)]['full_coors'])
            point_feat, img_pts_feat = self.p2img_mapping(pts_feat[coors_inv], point2img_index, batch_idx, points_img)
            pts_pred, _ = self.p2img_mapping(pts_pred_full[coors_inv], point2img_index, batch_idx, points_img)
            
            

        return batch_dict
    
    def voxelize_labels(labels, full_coors):
        lbxyz = torch.cat([labels.reshape(-1, 1), full_coors], dim=-1)
        unq_lbxyz, count = torch.unique(lbxyz, return_counts=True, dim=0)
        inv_ind = torch.unique(unq_lbxyz[:, 1:], return_inverse=True, dim=0)[1]
        label_ind = torch_scatter.scatter_max(count, inv_ind)[1]
        labels = unq_lbxyz[:, 0][label_ind]
        return labels


    def p2img_mapping(self, pts_fea, p2img_idx, batch_idx, points_img): # point to image mapping
        # pts_fea: (N_points, C)
        # p2img_idx: 
        # batch_idx: (N_points)
        img_feat = []
        img_pts_feat = Variable(torch.zeros(int(batch_idx.max().item()+1), 256, 1024, self.hiden_size)).to(batch_idx.device)
        for b in range(int(batch_idx.max().item()+1)):
            img_feat.append(pts_fea[batch_idx == b][p2img_idx[b]])
            img_pts_feat[b, points_img[b][:, 0], points_img[b][:, 1], :] = pts_fea[batch_idx == b][p2img_idx[b]]

        return torch.cat(img_feat, 0), img_pts_feat

    def img2p_mapping(self, pts_fea, p2img_idx, batch_idx, points_img, img_latent_full): # image vit features to 
        img_latent_full = torch.nn.functional.interpolate(img_latent_full[:, 1:, :].reshape(int(batch_idx.max().item()+1), 16, 64, -1).permute(0, 3, 1, 2).contiguous(), 
                                                          size=(256, 1024), mode='bilinear').permute(0, 2, 3, 1).contiguous()
        
        pts_with_img_feat = []
        for b in range(int(batch_idx.max().item()+1)):
            pts_batch = pts_fea[batch_idx == b] 
            pts_batch[p2img_idx[b]] += img_latent_full[b, points_img[b][:, 0], points_img[b][:, 1], :]
            pts_with_img_feat.append(pts_batch)
        
        return torch.cat(pts_with_img_feat, 0)