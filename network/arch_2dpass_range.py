import torch
import torch_scatter
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from network.basic_block import Lovasz_loss
from network.baseline import get_model_range as SPVCNN
from network.base_model import LightningBaseModel
from network.basic_block import ResNetFCN, DepthResNetFCN, ResNet_Model

from network.baseline import criterion

class xModalKD(nn.Module):
    def __init__(self,config):
        super(xModalKD, self).__init__()
        self.hiden_size = config['model_params']['hiden_size']
        self.scale_list = config['model_params']['scale_list']
        self.num_classes = config['model_params']['num_classes']
        self.lambda_xm = config['train_params']['lambda_xm']
        self.lambda_seg2d = config['train_params']['lambda_seg2d']
        self.num_scales = len(self.scale_list)

        self.multihead_3d_classifier = nn.ModuleList()
        for i in range(self.num_scales):
            self.multihead_3d_classifier.append(
                nn.Sequential(
                    nn.Linear(self.hiden_size, 128),
                    nn.ReLU(True),
                    nn.Linear(128, self.num_classes))
            )

        self.multihead_fuse_classifier_L, self.multihead_fuse_classifier_C = nn.ModuleList(), nn.ModuleList()
        for i in range(self.num_scales):
            self.multihead_fuse_classifier_L.append(
                nn.Sequential(
                    nn.Linear(self.hiden_size, 128),
                    nn.ReLU(True),
                    nn.Linear(128, self.num_classes))
            )
            self.multihead_fuse_classifier_C.append(
                nn.Sequential(
                    nn.Linear(self.hiden_size, 128),
                    nn.ReLU(True),
                    nn.Linear(128, self.num_classes))
            )
        self.leaners_L2C, self.leaners_C2L = nn.ModuleList(), nn.ModuleList()
        self.fcs1_L2C, self.fcs1_C2L = nn.ModuleList(), nn.ModuleList()
        self.fcs2_L2C, self.fcs2_C2L = nn.ModuleList(), nn.ModuleList()
        for i in range(self.num_scales):
            self.leaners_L2C.append(nn.Sequential(nn.Linear(self.hiden_size, self.hiden_size)))
            self.fcs1_L2C.append(nn.Sequential(nn.Linear(self.hiden_size * 2, self.hiden_size)))
            self.fcs2_L2C.append(nn.Sequential(nn.Linear(self.hiden_size, self.hiden_size)))
            self.leaners_C2L.append(nn.Sequential(nn.Linear(self.hiden_size, self.hiden_size)))
            self.fcs1_C2L.append(nn.Sequential(nn.Linear(self.hiden_size * 2, self.hiden_size)))
            self.fcs2_C2L.append(nn.Sequential(nn.Linear(self.hiden_size, self.hiden_size)))

        self.classifier_C = nn.Sequential(
            nn.Linear(self.hiden_size * self.num_scales, 128),
            nn.ReLU(True),
            nn.Linear(128, self.num_classes),
        )
        self.classifier_L = nn.Sequential(
            nn.Linear(self.hiden_size * self.num_scales, 128),
            nn.ReLU(True),
            nn.Linear(128, self.num_classes),
        )

        if 'seg_labelweights' in config['dataset_params']:
            seg_num_per_class = config['dataset_params']['seg_labelweights']
            seg_labelweights = seg_num_per_class / np.sum(seg_num_per_class)
            seg_labelweights = torch.Tensor(np.power(np.amax(seg_labelweights) / seg_labelweights, 1 / 3.0))
        else:
            seg_labelweights = None

        self.ce_loss = nn.CrossEntropyLoss(weight=seg_labelweights, ignore_index=config['dataset_params']['ignore_label'])
        self.lovasz_loss = Lovasz_loss(ignore=config['dataset_params']['ignore_label'])

        self.criterion = criterion(config)

    @staticmethod
    def p2img_mapping(pts_fea, p2img_idx, batch_idx):
        # pts_fea: (N_points, C)
        # p2img_idx: 
        # batch_idx: (N_points)
        img_feat = []
        for b in range(batch_idx.max()+1):
            img_feat.append(pts_fea[batch_idx == b][p2img_idx[b]])
        return torch.cat(img_feat, 0)

    @staticmethod
    def voxelize_labels(labels, full_coors):
        lbxyz = torch.cat([labels.reshape(-1, 1), full_coors], dim=-1)
        unq_lbxyz, count = torch.unique(lbxyz, return_counts=True, dim=0)
        inv_ind = torch.unique(unq_lbxyz[:, 1:], return_inverse=True, dim=0)[1]
        label_ind = torch_scatter.scatter_max(count, inv_ind)[1]
        labels = unq_lbxyz[:, 0][label_ind]
        return labels

    def seg_loss(self, logits, labels):
        ce_loss = self.ce_loss(logits, labels)
        lovasz_loss = self.lovasz_loss(F.softmax(logits, dim=1), labels)
        return ce_loss + lovasz_loss
    
    def fusion_to_single_KD(self, data_dict, idx):
        # # pts_feat: (N_pc, C)
        # # img_feat: (N_img, C)
        # # depth_feat: (N_img, C)
        # # proj_range_feat: (B, C, 64, 512)
        # # proj_xyz_feat: (B, C, 64, 512)
        # # proj_idx: (B, 1, 64, 512)
        # # laser_x: (N_points, 2)
        # # laser_y: (N_points, 2)
        # # batch_idx: (N_points)
        batch_idx = data_dict['batch_idx']
        point2img_index = data_dict['point2img_index']
        last_scale = self.scale_list[idx - 1] if idx > 0 else 1
        img_feat = data_dict['img_scale{}'.format(self.scale_list[idx])]
        pts_feat = data_dict['layer_{}'.format(idx)]['pts_feat']
        pts_feat_f = data_dict['layer_{}'.format(idx)]['pts_feat_f']
        coors_inv = data_dict['scale_{}'.format(last_scale)]['coors_inv']

        g_img_feat = data_dict['range_layer_{}'.format(idx)]

        laser_x = data_dict['laser_x']
        laser_y = data_dict['laser_y']
        batch_size = data_dict['batch_size']
        range_feats = []
        for batch_id in range(batch_size):
            x_batch = laser_x[torch.where(laser_x[:, 0] == batch_id)][:, -1]
            y_batch = laser_y[torch.where(laser_y[:, 0] == batch_id)][:, -1]
            range_feat = g_img_feat[batch_id, :, y_batch.long(), x_batch.long()]
            range_feats.append(range_feat.permute(1,0).contiguous())
        range_feats = torch.cat(range_feats, 0)

        # 3D prediction
        pts_pred_full = self.multihead_3d_classifier[idx](pts_feat)
        pts_pred = self.p2img_mapping(pts_pred_full[coors_inv], point2img_index, batch_idx)

        # correspondence
        pts_label_full = self.voxelize_labels(data_dict['labels'], data_dict['layer_{}'.format(idx)]['full_coors'])
        pts_feat_f2C = self.p2img_mapping(pts_feat_f, point2img_index, batch_idx)
        range_feats_2C = self.p2img_mapping(range_feats, point2img_index, batch_idx)

        # modality fusion: LiDAR to CAM
        feat_learner = F.relu(self.leaners_L2C[idx](pts_feat_f2C))
        feat_cat = torch.cat([range_feats_2C, feat_learner], 1)
        feat_cat = self.fcs1_L2C[idx](feat_cat)
        feat_weight = torch.sigmoid(self.fcs2_L2C[idx](feat_cat))
        fuse_feat_C = F.relu(feat_cat * feat_weight)
        # fusion prediction
        fuse_pred_C = self.multihead_fuse_classifier_C[idx](fuse_feat_C)

        # modality fusion:  CAM to LiDAR
        feat_learner = F.relu(self.leaners_C2L[idx](range_feats))
        feat_cat = torch.cat([pts_feat_f, feat_learner], 1)
        feat_cat = self.fcs1_C2L[idx](feat_cat)
        feat_weight = torch.sigmoid(self.fcs2_C2L[idx](feat_cat))
        fuse_feat_L = F.relu(feat_cat * feat_weight)

        # Segmentation Loss
        seg_loss_3d = self.seg_loss(pts_pred_full, pts_label_full)
        seg_loss_2d = self.seg_loss(fuse_pred_C, data_dict['img_label'])
        loss = seg_loss_3d + seg_loss_2d * self.lambda_seg2d / self.num_scales 

        # range - color - depth:
        mse_loss = nn.MSELoss()
        g_loss = mse_loss(range_feats_2C, img_feat)        
        loss += g_loss * self.lambda_seg2d / self.num_scales / img_feat.shape[0]

        # KL divergence
        xm_loss = F.kl_div(
            F.log_softmax(pts_pred, dim=1),
            F.softmax(fuse_pred_C.detach(), dim=1),
        )
        loss += xm_loss * self.lambda_xm / self.num_scales

        return loss, fuse_feat_C, fuse_feat_L



    def forward(self, data_dict):
        loss = 0
        img_seg_feat = []
        pts_seg_feat = []

        for idx in range(self.num_scales):
            singlescale_loss, fuse_feat_C, fuse_feat_L = self.fusion_to_single_KD(data_dict, idx)
            img_seg_feat.append(fuse_feat_C)
            pts_seg_feat.append(fuse_feat_L)
            loss += singlescale_loss

        img_seg_logits = self.classifier_C(torch.cat(img_seg_feat, 1))
        pts_seg_logits = self.classifier_L(torch.cat(pts_seg_feat, 1))
        loss += self.seg_loss(img_seg_logits, data_dict['img_label'])
        data_dict['logits'] = pts_seg_logits

        data_dict = self.criterion(data_dict)
        
        data_dict['loss'] += loss

        return data_dict


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

        self.model_3d = SPVCNN(backbone=config.model_params.backbone_2d,
                                pretrained=config.model_params.pretrained2d,
                                config=config)       
        
        self.model_2d = DepthResNetFCN(
                backbone=config.model_params.backbone_2d,
                pretrained=config.model_params.pretrained2d,
                config=config
            )
        
        self.fusion = xModalKD(config)
        
    def forward(self, data_dict):
        # 3D network
        data_dict = self.model_3d(data_dict)

        # training with 2D network
        data_dict = self.model_2d(data_dict)
        data_dict = self.fusion(data_dict)

        return data_dict