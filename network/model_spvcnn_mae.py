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

        self.model_3d = SPVCNN_MAE(config['model_params']['mae_parameters'])
        
        # decoder layer
        self.classifier_3d = nn.Sequential(
            nn.Linear(self.hiden_size * self.num_scales, 128),
            nn.ReLU(True),
            nn.Linear(128, self.num_classes),
        )
        # loss
        self.criterion = criterion(config)

        self.model_2d = ResNetFCN(
            backbone=config.model_params.backbone_2d,
            pretrained=config.model_params.pretrained2d,
            config=config
        )

        self.fusion = xModalKD(config)
        
        self._init_pretrain_()
        self._init_mae_(True)
        
    def _init_pretrain_(self):
        model_checkpoint = torch.load('/home/yanqiao/2DPASS/pretrained/semantickitti/2DPASS_4scale_64dim/best_model.ckpt')
        model_state_dict = model_checkpoint['state_dict']
        for param in self.state_dict():
            model_param = param
            if model_param in model_state_dict and self.state_dict()[param].size() == model_state_dict[model_param].size():
                self.state_dict()[param] = model_state_dict[model_param]
                print('initing with param: ', model_param)

    def _init_mae_(self, pretrain=False):
        if pretrain:
            self.model_3d.load_params_from_file('/home/yanqiao/2DPASS/pretrained/spconv_mae/checkpoint_epoch_100.pth', None)
            print('Loaded MAE from pre-trained weights')
            # for param in self.model_3d.parameters():
            #     param.requires_grad = True
            # for param in self.model_3d.image_encoder.parameters():
            #     param.requires_grad = False
        else:
            print('vanilla training !')
            
    def forward(self, batch_dict):
        raw_images = batch_dict['img']
        batch_dict = self.model_2d(batch_dict)
        sample_points_batch_idx = batch_dict['sample_points_batch_idx'][:, 0]
        raw_images = batch_dict['img']
        sample_index = batch_dict['sample_index']
        batch_dict['spconv_points'] = batch_dict['points']

        # B = batch_dict['batch_size']
        Batch_size, _, H_raw, W_raw = raw_images.size() # (256, 1024)
        images = torch.nn.functional.interpolate(raw_images, size=self.img_size, mode='bilinear')

        # color image encoding
        img_latent, img_mask, img_ids_restore, _ = self.model_3d.image_encoder.forward_encoder(images, self.img_mask_ratio)
        # img_latent_full, img_mask_full, img_ids_restore_full = self.image_encoder.forward_encoder(images, 0)
        img_latent_full = self.model_3d.forward_decoder_img(img_latent, img_ids_restore)
        img_latent_full = self.model_3d.img_conv(img_latent_full.reshape(Batch_size, self.img_size[0]//self.model_3d.scale_factor[0], self.img_size[1]//self.model_3d.scale_factor[1], -1).permute(0, 3, 1, 2).contiguous())
        # img_latent_full: (B, C, H, W)

        batch_dict['img_latent_full'] = img_latent_full

        # spvcnn:
        batch_dict = self.model_3d.pc_encoder(batch_dict)

        batch_dict = self.fusion(batch_dict)
        output = batch_dict['output_3d']
        batch_dict['logits'] = self.classifier_3d(output)

        
        batch_dict = self.criterion(batch_dict)
        

        return batch_dict
    


class xModalKD(nn.Module):
    def __init__(self,config):
        super(xModalKD, self).__init__()
        self.hiden_size = config['model_params']['hiden_size']
        self.scale_list = config['model_params']['scale_list']
        self.num_classes = config['model_params']['num_classes']
        self.lambda_xm = config['train_params']['lambda_xm']
        self.lambda_seg2d = config['train_params']['lambda_seg2d']
        self.num_scales = len(self.scale_list)

        self.img_size = (256, 1024)

        self.multihead_3d_classifier = nn.ModuleList()
        for i in range(self.num_scales):
            self.multihead_3d_classifier.append(
                nn.Sequential(
                    nn.Linear(self.hiden_size, 128),
                    nn.ReLU(True),
                    nn.Linear(128, self.num_classes))
            )

        self.multihead_fuse_classifier = nn.ModuleList()
        for i in range(self.num_scales):
            self.multihead_fuse_classifier.append(
                nn.Sequential(
                    nn.Linear(self.hiden_size, 128),
                    nn.ReLU(True),
                    nn.Linear(128, self.num_classes))
            )
        self.leaners = nn.ModuleList()
        self.fcs1 = nn.ModuleList()
        self.fcs2 = nn.ModuleList()
        for i in range(self.num_scales):
            self.leaners.append(nn.Sequential(nn.Linear(self.hiden_size, self.hiden_size)))
            self.fcs1.append(nn.Sequential(nn.Linear(self.hiden_size * 2, self.hiden_size)))
            self.fcs2.append(nn.Sequential(nn.Linear(self.hiden_size, self.hiden_size)))

        self.classifier = nn.Sequential(
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
    def seg_loss(self, logits, labels):
        ce_loss = self.ce_loss(logits, labels)
        lovasz_loss = self.lovasz_loss(F.softmax(logits, dim=1), labels)
        return ce_loss + lovasz_loss
    
    def p2img_mapping_spsample(self, pts_fea, p2img_idx, batch_idx, pts_fea_sample, batch_idx_sample, sample_index, img_latent_full, points_img):
        # pts_fea: (N_points, C)
        # p2img_idx: 
        # batch_idx: (N_points)
        # img_latent_full: B, C, H, W
        pts_feats = []
        pts_feats_cls = []
        img_pts_feat = Variable(torch.zeros(int(batch_idx.max().item()+1), self.img_size[0], self.img_size[1], self.hiden_size)).to(batch_idx.device)

        img_latent_full = img_latent_full.permute(0,2,3,1).contiguous()
        
        for b in range(int(batch_idx.max().item()+1)):
            pts_batch = pts_fea[batch_idx == b]
            pts_sample_points_batch = pts_fea_sample[batch_idx_sample == b]

            pts_batch_new = Variable(pts_batch.new_zeros(pts_batch.size()))
            pts_batch_new_cls = Variable(pts_batch.new_ones(pts_batch.shape[0], 1))
            pts_img_batch_new = Variable(pts_batch.new_zeros(pts_batch.shape[0], self.hiden_size))
            # print('sample_index b', sample_index[b].size())
            # print('pts_sample_points_batch', pts_sample_points_batch.size())
            pts_batch_new[sample_index[b]] = pts_sample_points_batch
            pts_batch_new_cls[sample_index[b]] -= 1
            pts_img_batch_new[p2img_idx[b]] = img_latent_full[b, points_img[b][:, 0], points_img[b][:, 1], :]  

            img_pts_feat[b, points_img[b][:, 0], points_img[b][:, 1], :] = pts_batch_new[p2img_idx[b]]

            pts_batch_new += pts_img_batch_new

            pts_feats.append(pts_batch_new)
            pts_feats_cls.append(pts_batch_new_cls)

        return torch.cat(pts_feats, 0), torch.cat(pts_feats_cls, 0), img_pts_feat
    
    def voxelize_labels(self, labels, full_coors):
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
        img_pts_feat = Variable(torch.zeros(int(batch_idx.max().item()+1), 256, 1024, pts_fea.shape[-1])).to(batch_idx.device)
        for b in range(int(batch_idx.max().item()+1)):
            img_feat.append(pts_fea[batch_idx == b][p2img_idx[b]])
            img_pts_feat[b, points_img[b][:, 0], points_img[b][:, 1], :] = pts_fea[batch_idx == b][p2img_idx[b]]

        return torch.cat(img_feat, 0), img_pts_feat


    def forward(self, batch_dict):
        sample_points_batch_idx = batch_dict['sample_points_batch_idx'][:, 0]
        sample_index = batch_dict['sample_index']
        batch_dict['spconv_points'] = batch_dict['points']
        img_latent_full = batch_dict['img_latent_full']
        loss = 0
        img_seg_feat = []
        enc_feats = []

        for idx in range(4):
            last_scale = self.scale_list[idx - 1] if idx > 0 else 1
            img_feat = batch_dict['img_scale{}'.format(self.scale_list[idx])]
            points_img = batch_dict['img_indices']
            point2img_index = batch_dict['point2img_index'] # list # (N_pc2img, N_pc2img, ..., ... )
            batch_idx = batch_dict['batch_idx']
            pts_feat_f = batch_dict['spconv_points_layer_{}'.format(idx)]['pts_feat_f']
            pts_feat = batch_dict['spconv_points_layer_{}'.format(idx)]['pts_feat']
            coors_inv = batch_dict['spconv_points_scale_{}'.format(last_scale)]['coors_inv']

            pts_sample_feat_f = pts_feat_f

            # batch_idx: (N)
            # pts_feat_f: (N, 64)

            # process pts with vit image feat
            sample_point_feat, sample_point_feat_f_cls, img_pts_feat = self.p2img_mapping_spsample(pts_feat[coors_inv], point2img_index, batch_idx, pts_sample_feat_f, sample_points_batch_idx, sample_index, img_latent_full, points_img)
            sample_point_feat_f, sample_point_feat_f_cls, img_pts_feat = self.p2img_mapping_spsample(pts_feat_f, point2img_index, batch_idx, pts_sample_feat_f, sample_points_batch_idx, sample_index, img_latent_full, points_img)
            pts_feat[coors_inv] = sample_point_feat

            # 3D prediction
            pts_pred_full = self.multihead_3d_classifier[idx](pts_feat)
            pts_feat_f = sample_point_feat_f
            
            # correspondence
            pts_label_full = self.voxelize_labels(batch_dict['labels'], batch_dict['spconv_points_layer_{}'.format(idx)]['full_coors'])
            pts_feat, _ = self.p2img_mapping(pts_feat[coors_inv], point2img_index, batch_idx, points_img)
            pts_pred, _ = self.p2img_mapping(pts_pred_full[coors_inv], point2img_index, batch_idx, points_img)
            
            # modality fusion
            feat_learner = F.relu(self.leaners[idx](pts_feat))
            feat_cat = torch.cat([img_feat, feat_learner], 1)
            feat_cat = self.fcs1[idx](feat_cat)
            feat_weight = torch.sigmoid(self.fcs2[idx](feat_cat))
            fuse_feat = F.relu(feat_cat * feat_weight)

            # fusion prediction
            fuse_pred = self.multihead_fuse_classifier[idx](fuse_feat)
            
            # Segmentation Loss
            seg_loss_3d = self.seg_loss(pts_pred_full, pts_label_full)
            seg_loss_2d = self.seg_loss(fuse_pred, batch_dict['img_label'])
            loss = seg_loss_3d + seg_loss_2d * self.lambda_seg2d / self.num_scales

            # KL divergence
            xm_loss = F.kl_div(
                F.log_softmax(pts_pred, dim=1),
                F.softmax(fuse_pred.detach(), dim=1),
            )
            loss += xm_loss * self.lambda_xm / self.num_scales

            img_seg_feat.append(fuse_feat)
            enc_feats.append(pts_feat_f)
        
        img_seg_logits = self.classifier(torch.cat(img_seg_feat, 1))
        loss += self.seg_loss(img_seg_logits, batch_dict['img_label']) 
        output = torch.cat(enc_feats, dim=1)
        batch_dict['output_3d'] = output
        batch_dict['loss'] = loss

        return batch_dict
        