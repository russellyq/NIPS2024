import os

import torch
import torch.nn as nn
import numpy as np

from pcdet.utils.spconv_utils import find_all_spconv_keys
from network.model_mae import mae_vit_base_patch16, mae_vit_large_patch16, mae_vit_huge_patch14, mae_vit_large_patch8, MaskedAutoencoderViT
from torch.autograd import Variable



import torch
import torch_scatter
import spconv.pytorch as spconv
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
import numpy as np
from pcdet.utils.voxel_feat_generation import voxel_3d_generator, voxelization
from pcdet.models.backbones_3d.spconv_spvcnn import SparseBasicBlock
import os
import torch
import torch.nn as nn
import numpy as np
from functools import partial
from timm.models.vision_transformer import Block
from pcdet.datasets.kitti.laserscan import LaserScan
from torch.autograd import Variable

class point_encoder(nn.Module):
    def __init__(self, in_channels, out_channels, scale):
        super(point_encoder, self).__init__()
        self.scale = scale
        self.layer_in = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.LeakyReLU(0.1, True),
        )
        self.PPmodel = nn.Sequential(
            nn.Linear(in_channels, out_channels // 2),
            nn.LeakyReLU(0.1, True),
            nn.BatchNorm1d(out_channels // 2),
            nn.Linear(out_channels // 2, out_channels // 2),
            nn.LeakyReLU(0.1, True),
            nn.BatchNorm1d(out_channels // 2),
            nn.Linear(out_channels // 2, out_channels),
            nn.LeakyReLU(0.1, True),
        )
        self.layer_out = nn.Sequential(
            nn.Linear(2 * out_channels, out_channels),
            nn.LeakyReLU(0.1, True),
            nn.Linear(out_channels, out_channels))

    @staticmethod
    def downsample(coors, p_fea, scale=2):
        batch = coors[:, 0:1]
        coors = coors[:, 1:] // scale
        inv = torch.unique(torch.cat([batch, coors], 1), return_inverse=True, dim=0)[1]
        return torch_scatter.scatter_mean(p_fea, inv, dim=0), inv

    def forward(self, features, data_dict, sample_points=False):
        if sample_points:
            output, inv = self.downsample(data_dict['sample_points_coors'], features)
            identity = self.layer_in(features)
            output = self.PPmodel(output)[inv]
            output = torch.cat([identity, output], dim=1)

            v_feat = torch_scatter.scatter_mean(
                self.layer_out(output[data_dict['sample_points_coors_inv']]),
                data_dict['sample_points_scale_{}'.format(self.scale)]['coors_inv'],
                dim=0
            )
            data_dict['sample_points_coors'] = data_dict['sample_points_scale_{}'.format(self.scale)]['coors']
            data_dict['sample_points_coors_inv'] = data_dict['sample_points_scale_{}'.format(self.scale)]['coors_inv']
            data_dict['sample_points_full_coors'] = data_dict['sample_points_scale_{}'.format(self.scale)]['full_coors']
        else:
            output, inv = self.downsample(data_dict['spconv_points_coors'], features)
            identity = self.layer_in(features)
            output = self.PPmodel(output)[inv]
            output = torch.cat([identity, output], dim=1)

            v_feat = torch_scatter.scatter_mean(
                self.layer_out(output[data_dict['spconv_points_coors_inv']]),
                data_dict['spconv_points_scale_{}'.format(self.scale)]['coors_inv'],
                dim=0
            )
            data_dict['spconv_points_coors'] = data_dict['spconv_points_scale_{}'.format(self.scale)]['coors']
            data_dict['spconv_points_coors_inv'] = data_dict['spconv_points_scale_{}'.format(self.scale)]['coors_inv']
            data_dict['spconv_points_full_coors'] = data_dict['spconv_points_scale_{}'.format(self.scale)]['full_coors']

        
        return v_feat


class SPVBlock(nn.Module):
    def __init__(self, in_channels, out_channels, indice_key, scale, last_scale, spatial_shape, sample_points=False):
        super(SPVBlock, self).__init__()
        self.scale = scale
        self.indice_key = indice_key
        self.layer_id = indice_key.split('_')[1]
        self.last_scale = last_scale
        self.spatial_shape = spatial_shape
        self.v_enc = spconv.SparseSequential(
            SparseBasicBlock(in_channels, out_channels, self.indice_key),
            SparseBasicBlock(out_channels, out_channels, self.indice_key),
        )
        self.sample_points = sample_points
        
        self.p_enc = point_encoder(in_channels, out_channels, scale)

    def forward(self, data_dict):
        coors_inv_last = data_dict['spconv_points_scale_{}'.format(self.last_scale)]['coors_inv']
        coors_inv = data_dict['spconv_points_scale_{}'.format(self.scale)]['coors_inv']

        # voxel encoder
        v_fea = self.v_enc(data_dict['spconv_points_sparse_tensor'])
        data_dict['spconv_points_layer_{}'.format(self.layer_id)] = {}
        data_dict['spconv_points_layer_{}'.format(self.layer_id)]['pts_feat'] = v_fea.features
        data_dict['spconv_points_layer_{}'.format(self.layer_id)]['full_coors'] = data_dict['spconv_points_full_coors']
        v_fea_inv = torch_scatter.scatter_mean(v_fea.features[coors_inv_last], coors_inv, dim=0)

        # point encoder
        p_fea = self.p_enc(
            features=data_dict['spconv_points_sparse_tensor'].features+v_fea.features,
            data_dict=data_dict
        )

        # fusion and pooling
        data_dict['spconv_points_sparse_tensor'] = spconv.SparseConvTensor(
            features=p_fea+v_fea_inv,
            indices=data_dict['spconv_points_coors'],
            spatial_shape=self.spatial_shape,
            batch_size=data_dict['batch_size']
        )
        data_dict['spconv_points_layer_{}'.format(self.layer_id)]['pts_feat_f'] = p_fea[coors_inv]

        if self.sample_points:
            coors_inv_last = data_dict['sample_points_scale_{}'.format(self.last_scale)]['coors_inv']
            coors_inv = data_dict['sample_points_scale_{}'.format(self.scale)]['coors_inv']

            # voxel encoder
            v_fea = self.v_enc(data_dict['sample_points_sparse_tensor'])
            data_dict['sample_points_layer_{}'.format(self.layer_id)] = {}
            data_dict['sample_points_layer_{}'.format(self.layer_id)]['pts_feat'] = v_fea.features
            data_dict['sample_points_layer_{}'.format(self.layer_id)]['full_coors'] = data_dict['sample_points_full_coors']
            v_fea_inv = torch_scatter.scatter_mean(v_fea.features[coors_inv_last], coors_inv, dim=0)

            # point encoder
            p_fea = self.p_enc(
                features=data_dict['sample_points_sparse_tensor'].features+v_fea.features,
                data_dict=data_dict,
                sample_points=True,
            )

            # fusion and pooling
            data_dict['sample_points_sparse_tensor'] = spconv.SparseConvTensor(
                features=p_fea+v_fea_inv,
                indices=data_dict['sample_points_coors'],
                spatial_shape=self.spatial_shape,
                batch_size=data_dict['batch_size']
            )
            data_dict['sample_points_layer_{}'.format(self.layer_id)]['pts_feat_f'] = p_fea[coors_inv]

        return data_dict

class SPVCNN_Decoder(nn.Module):
    def __init__(self, hidden_size=64):
        super(SPVCNN_Decoder, self).__init__()
        self.input_dims = 4
        self.hiden_size = hidden_size
        self.scale_list = [16, 8, 4, 2]
        self.num_scales = len(self.scale_list)
        min_volume_space = [-50, -50, -4]
        max_volume_space = [50, 50, 2]
        self.coors_range_xyz = [[min_volume_space[0], max_volume_space[0]],
                                [min_volume_space[1], max_volume_space[1]],
                                [min_volume_space[2], max_volume_space[2]]]
        self.spatial_shape = np.array([1000, 1000, 60])
        self.strides = [int(scale / self.scale_list[0]) for scale in self.scale_list]

        self.spv_enc = nn.ModuleList()
        for i in range(self.num_scales):
            self.spv_enc.append(SparseBasicBlock(
                in_channels=self.hiden_size * 4 + 1,
                out_channels=self.hiden_size * 4 + 1,
                indice_key='spv_decoder_' + str(i)
                )
            )
        self.pred = SparseBasicBlock(
                in_channels=self.hiden_size * 4 + 1,
                out_channels=4,
                indice_key='spv_decoder_pred'
                )
        print('succesfully build SPVCNN decoder')

    def forward(self, sample_point_feat_fs, sample_point_feat_fs_cls, batch_dict):

        sp_tensor = spconv.SparseConvTensor(
            features=torch.cat([sample_point_feat_fs_cls, sample_point_feat_fs], dim=1),
            indices=batch_dict['spconv_points_full_coors'].int(),
            spatial_shape=self.spatial_shape,
            batch_size=batch_dict['batch_size']
        )
        for i in range(self.num_scales):
            sp_tensor = self.spv_enc[i](sp_tensor)
        sp_tensor = self.pred(sp_tensor)
        return sp_tensor.features

class SPVCNN(nn.Module):
    def __init__(self, hidden_size=64, sample_points=False):
        super(SPVCNN, self).__init__()
        self.input_dims = 4
        self.hiden_size = hidden_size
        self.scale_list = [2, 4, 8, 16]
        self.num_scales = len(self.scale_list)
        min_volume_space = [-50, -50, -4]
        max_volume_space = [50, 50, 2]
        self.coors_range_xyz = [[min_volume_space[0], max_volume_space[0]],
                                [min_volume_space[1], max_volume_space[1]],
                                [min_volume_space[2], max_volume_space[2]]]
        self.spatial_shape = np.array([1000, 1000, 60])
        self.strides = [int(scale / self.scale_list[0]) for scale in self.scale_list]

        # voxelization
        self.voxelizer = voxelization(
            coors_range_xyz=self.coors_range_xyz,
            spatial_shape=self.spatial_shape,
            scale_list=self.scale_list,
            sample_points=sample_points
        )
        # input processing
        self.voxel_3d_generator = voxel_3d_generator(
            in_channels=self.input_dims,
            out_channels=self.hiden_size,
            coors_range_xyz=self.coors_range_xyz,
            spatial_shape=self.spatial_shape,
            sample_points=sample_points
        )
        # encoder layers
        self.spv_enc = nn.ModuleList()
        for i in range(self.num_scales):
            self.spv_enc.append(SPVBlock(
                in_channels=self.hiden_size,
                out_channels=self.hiden_size,
                indice_key='spv_'+ str(i),
                scale=self.scale_list[i],
                last_scale=self.scale_list[i-1] if i > 0 else 1,
                spatial_shape=np.int32(self.spatial_shape // self.strides[i])[::-1].tolist(),
                sample_points=sample_points
                )
            )
        print('succesfully build SPVCNN encoder')
    
    def forward(self, data_dict):
        with torch.no_grad():
            data_dict = self.voxelizer(data_dict)
        data_dict = self.voxel_3d_generator(data_dict)
        for i in range(self.num_scales):
            data_dict = self.spv_enc[i](data_dict)       
        return data_dict

class SPVCNN_MAE(nn.Module):
    def __init__(self, model_cfg, num_class, dataset, sample_points=True):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.dataset = dataset
        self.register_buffer('global_step', torch.LongTensor(1).zero_())

        self.module_topology = [
            'vfe', 'backbone_3d', 'map_to_bev_module', 'pfe',
            'backbone_2d', 'dense_head',  'point_head', 'roi_head'
        ]
        self.input_dims = 4
        self.hiden_size = 64
        self.scale_list = [2, 4, 8, 16]
        self.num_scales = len(self.scale_list)
        min_volume_space = [-50, -50, -4]
        max_volume_space = [50, 50, 2]

        self.img_mask_ratio =  0.75
        self.img_size = (256, 1024)
        self.embed_dim = 1024
        self.decoder_embed_dim = 512

        self.pc_encoder = SPVCNN(hidden_size=self.hiden_size, sample_points=sample_points)
        # self.image_encoder = mae_vit_large_patch16(in_chans=3, img_with_size=self.img_size, out_chans=3, with_patch_2d=False)
        
        self.image_encoder = MaskedAutoencoderViT(in_chans=3, img_with_size=self.img_size, out_chans=3, 
                                                    patch_size=16, embed_dim=self.embed_dim, depth=24, num_heads=16,
                                                    decoder_embed_dim=self.decoder_embed_dim, decoder_depth=8, decoder_num_heads=16,
                                                    mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), with_patch_2d=False)
        
        self.maxpool = nn.MaxPool2d(kernel_size=16, stride=16)
        self.pts_decode = nn.Linear(self.hiden_size * 4, self.decoder_embed_dim, bias=True)

        self.img_conv = nn.Linear(self.embed_dim, self.hiden_size, bias=True)

        self.pc_decoder = SPVCNN_Decoder(hidden_size=self.hiden_size)

        print('succesfully build SPVCNN_MAE model')


    @property
    def mode(self):
        return 'TRAIN' if self.training else 'TEST'

    def update_global_step(self):
        self.global_step += 1
    
    def p2img_mapping(self, pts_fea, p2img_idx, batch_idx, points_img):
        # pts_fea: (N_points, C)
        # p2img_idx: 
        # batch_idx: (N_points)
        img_feat = []
        img_pts_feat = Variable(torch.zeros(int(batch_idx.max().item()+1), 256, 1024, self.hiden_size)).to(batch_idx.device)
        
        for b in range(int(batch_idx.max().item()+1)):
            
            img_feat.append(pts_fea[batch_idx == b][p2img_idx[b]])

            img_pts_feat[b, points_img[b][:, 0], points_img[b][:, 1], :] = pts_fea[batch_idx == b][p2img_idx[b]]

        return torch.cat(img_feat, 0), img_pts_feat

    # binary mask: 0 is keep, 1 is remove
    def p2img_mapping_spsample(self, pts_fea, p2img_idx, batch_idx, pts_fea_sample, batch_idx_sample, sample_index, img_latent_full, points_img):
        # pts_fea: (N_points, C)
        # p2img_idx: 
        # batch_idx: (N_points)
        pts_feats = []
        pts_feats_cls = []
        # img_feats = []
        img_latent_full = torch.nn.functional.interpolate(img_latent_full[:, 1:, :].reshape(int(batch_idx.max().item()+1), 16, 64, -1).permute(0, 3, 1, 2).contiguous(), 
                                                          size=(256, 1024), mode='bilinear').permute(0, 2, 3, 1).contiguous()

        for b in range(int(batch_idx.max().item()+1)):
            pts_batch = pts_fea[batch_idx == b]
            pts_img_batch_new = Variable(pts_batch.new_zeros(pts_batch.shape[0], self.hiden_size))
            pts_batch_new = Variable(pts_batch.new_zeros(pts_batch.size()))
            pts_batch_new_cls = Variable(pts_batch.new_ones(pts_batch.shape[0], 1))
            pts_sample_points_batch = pts_fea_sample[batch_idx_sample == b]
            pts_batch_new[sample_index[b]] = pts_sample_points_batch
            pts_batch_new_cls[sample_index[b]] -= 1

            
            pts_img_batch_new[p2img_idx[b]] = img_latent_full[b, points_img[b][:, 0], points_img[b][:, 1], :]  
            pts_batch_new += pts_img_batch_new
            
            # img_feats.append( pts_img_batch_new)
            pts_feats.append(pts_batch_new)
            pts_feats_cls.append(pts_batch_new_cls)

        return torch.cat(pts_feats, 0), torch.cat(pts_feats_cls, 0)
    
    def forward(self, batch_dict):
        batch_dict['batch_idx'] = batch_dict['points'][:, 0]

        range_loss, img_loss = 0, 0
        loss = range_loss + img_loss

        # raw image & range image shape: (B, 3, 376, 1241)
        laser_range_in = batch_dict['laser_range_in'].permute(0,3,1,2).contiguous()
        laser_x = batch_dict['laser_x']
        laser_y = batch_dict['laser_y']
        raw_images = batch_dict['images']
        # B = batch_dict['batch_size']

        # print('img size: ', raw_images.size())

        Batch_size, _, H_raw, W_raw = raw_images.size() # (256, 1024)

        images = torch.nn.functional.interpolate(raw_images, size=self.img_size, mode='bilinear')

        # color image encoding
        img_latent, img_mask, img_ids_restore = self.image_encoder.forward_encoder(images, self.img_mask_ratio)
        img_latent_full, img_mask_full, img_ids_restore_full = self.image_encoder.forward_encoder(images, 0)
        
        img_latent_full = self.img_conv(img_latent_full)
        # img_latent: (B, L=H*W / 16 / 16 * (1-mask_ratio) + 1=256+1=257, C=1024)
        # img_mask: (B, L=H*W/16/16=1024)   # 0 is keep, 1 is remove
        # img_ids_restore: (B, L=H*W/16/16=1024)
                
        batch_dict = self.pc_encoder(batch_dict)
        # from IPython import embed; embed()

        point_feat_fs = []
        sample_point_feat_fs = []

        image_pts_feats = []

        # process spcov features to image
        for idx in range(4):

            # spoconv_points
            # sp features to image
            points_img = batch_dict['points_img']
            point2img_index = batch_dict['point2img_index'] # list # (N_pc2img, N_pc2img, ..., ... )
            spconv_points_batch_idx = batch_dict['spconv_points_batch_idx'][:, 0]
            pts_feat_f = batch_dict['spconv_points_layer_{}'.format(idx)]['pts_feat_f']
            point_feat_f, img_pts_feat = self.p2img_mapping(pts_feat_f, point2img_index, spconv_points_batch_idx, points_img)
            # ( N_PC2img_ba1 + N_PC2img_ba2 + ... ... , 64)
            point_feat_fs.append(point_feat_f)
            image_pts_feats.append(img_pts_feat)

            # print('point_feat_f:', point_feat_f.size()) # (N_PC2img_ba1 + N_PC2img_ba2 + ... ..., 64)
            # print('image_pts_feats:', img_pts_feat.size()) # (B, 256, 1024, 64)

            # sample points
            point2img_index = batch_dict['point2img_index'] # list # (N_pc2img, N_pc2img, ..., ... )
            sample_points_batch_idx = batch_dict['sample_points_batch_idx'][:, 0]
            pts_sample_feat_f = batch_dict['sample_points_layer_{}'.format(idx)]['pts_feat_f']
            sample_index = batch_dict['sample_index']
            sample_point_feat_f, sample_point_feat_f_cls = self.p2img_mapping_spsample(pts_feat_f, point2img_index, spconv_points_batch_idx, pts_sample_feat_f, sample_points_batch_idx, sample_index, img_latent_full, points_img)
            # ( N_PC2img_ba1 + N_PC2img_ba2 + ... ... , 64)
            sample_point_feat_fs.append(sample_point_feat_f)

            # print('sample_point_feat_f:', sample_point_feat_f.size()) # (N_PC2img_ba1 + N_PC2img_ba2 + ... ..., 64)

            
        # spconv points to image
        image_pts_feats = torch.cat(image_pts_feats, -1).permute(0, 3, 1, 2).contiguous()
        image_pts_feats = self.maxpool(image_pts_feats).permute(0, 2, 3, 1).contiguous()

        image_pts_feats = image_pts_feats.reshape(Batch_size, -1, self.hiden_size * 4)
        image_pts_feats = self.pts_decode(image_pts_feats)

        img_pred = self.forward_decoder(img_latent, img_ids_restore, image_pts_feats,
                                        self.image_encoder.decoder_embed,
                                        self.image_encoder.mask_token,
                                        self.image_encoder.decoder_pos_embed,
                                        self.image_encoder.decoder_blocks,
                                        self.image_encoder.decoder_norm,
                                        self.image_encoder.decoder_pred)
        img_loss = self.image_encoder.forward_loss(images, img_pred, img_mask)

        # sample points + image points
        sample_point_feat_fs = torch.cat(sample_point_feat_fs, 1)
        sp_tensor_pred = self.pc_decoder(sample_point_feat_fs, sample_point_feat_f_cls, batch_dict)

        range_loss = self.spconv_loss(sp_tensor_pred, batch_dict['points'][:, 1:], sample_point_feat_f_cls, batch_dict)

        loss = range_loss + img_loss
        if self.training:
            tb_dict = {
                'range_loss': range_loss,
                'img_loss': img_loss,
            }
            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, {}
        else:
            
            # inference at batch_size=1
            pred_dicts, recall_dicts = {}, {}
    
    def spconv_loss(self, pred, points, mask, batch_dict):
        batch_idx = batch_dict['batch_idx']
        sample_index = batch_dict['unsample_index']
        pred_unsamples, points_unsamples = [], []
        for b in range(int(batch_idx.max().item()+1)):
            pred_batch = pred[batch_idx == b]
            pred_unsample = pred_batch[sample_index[b]]
            pred_unsamples.append(pred_unsample)

            points_batch = points[batch_idx == b]
            points_unsample = points_batch[sample_index[b]]
            points_unsamples.append(points_unsample)
        pred_unsamples = torch.cat(pred_unsamples, 0)
        points_unsamples = torch.cat(points_unsamples, 0)

        loss = (pred_unsamples - points_unsamples) ** 2
        loss = loss.mean(dim=-1)
        loss = loss.sum() / mask.sum()
        return loss
    
    def forward_decoder(self, x, ids_restore, x_mm, \
                        decoder_embed, \
                        mask_token, \
                        decoder_pos_embed,\
                        decoder_blocks, \
                        decoder_norm,\
                        decoder_pred):# x, ids_restore: (B, L1=H*W/16/16*(1-mask_ratio)+1=257, C) , (B, L2=H*W/16/16=1024)
        # embed tokens
        x = decoder_embed(x) # (B, L1, D=512)

        # append mask tokens to sequence
        mask_tokens = mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1) # (B, L2+1-L1=1024-256=768, D)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token # (B, L2, D)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle # (B, L2, D)
        
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token # (B, L2+1, D)

        # add pos embed
        x = x + decoder_pos_embed # (B, L2+1, D)

        x[:, 1:, :] = x[:, 1:, :] + x_mm
        
        # apply Transformer blocks
        for blk in decoder_blocks:
            x = blk(x)
        # x: (B, L2+1, D)
        x = decoder_norm(x)

        # predictor projection
        x = decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x


    def _load_state_dict(self, model_state_disk, *, strict=True):
        state_dict = self.state_dict()  # local cache of state_dict

        spconv_keys = find_all_spconv_keys(self)

        update_model_state = {}
        for key, val in model_state_disk.items():
            if key in spconv_keys and key in state_dict and state_dict[key].shape != val.shape:
                # with different spconv versions, we need to adapt weight shapes for spconv blocks
                # adapt spconv weights from version 1.x to version 2.x if you used weights from spconv 1.x

                val_native = val.transpose(-1, -2)  # (k1, k2, k3, c_in, c_out) to (k1, k2, k3, c_out, c_in)
                if val_native.shape == state_dict[key].shape:
                    val = val_native.contiguous()
                else:
                    assert val.shape.__len__() == 5, 'currently only spconv 3D is supported'
                    val_implicit = val.permute(4, 0, 1, 2, 3)  # (k1, k2, k3, c_in, c_out) to (c_out, k1, k2, k3, c_in)
                    if val_implicit.shape == state_dict[key].shape:
                        val = val_implicit.contiguous()

            if key in state_dict and state_dict[key].shape == val.shape:
                update_model_state[key] = val
                # logger.info('Update weight %s: %s' % (key, str(val.shape)))

        if strict:
            self.load_state_dict(update_model_state)
        else:
            state_dict.update(update_model_state)
            self.load_state_dict(state_dict)
        return state_dict, update_model_state

    def load_params_from_file(self, filename, logger, to_cpu=False, pre_trained_path=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['model_state']
        if not pre_trained_path is None:
            pretrain_checkpoint = torch.load(pre_trained_path, map_location=loc_type)
            pretrain_model_state_disk = pretrain_checkpoint['model_state']
            model_state_disk.update(pretrain_model_state_disk)
            
        version = checkpoint.get("version", None)
        if version is not None:
            logger.info('==> Checkpoint trained from version: %s' % version)

        state_dict, update_model_state = self._load_state_dict(model_state_disk, strict=False)

        for key in state_dict:
            if key not in update_model_state:
                logger.info('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))

        logger.info('==> Done (loaded %d/%d)' % (len(update_model_state), len(state_dict)))

    def load_params_with_optimizer(self, filename, to_cpu=False, optimizer=None, logger=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        epoch = checkpoint.get('epoch', -1)
        it = checkpoint.get('it', 0.0)

        self._load_state_dict(checkpoint['model_state'], strict=True)

        if optimizer is not None:
            if 'optimizer_state' in checkpoint and checkpoint['optimizer_state'] is not None:
                logger.info('==> Loading optimizer parameters from checkpoint %s to %s'
                            % (filename, 'CPU' if to_cpu else 'GPU'))
                optimizer.load_state_dict(checkpoint['optimizer_state'])
            else:
                assert filename[-4] == '.', filename
                src_file, ext = filename[:-4], filename[-3:]
                optimizer_filename = '%s_optim.%s' % (src_file, ext)
                if os.path.exists(optimizer_filename):
                    optimizer_ckpt = torch.load(optimizer_filename, map_location=loc_type)
                    optimizer.load_state_dict(optimizer_ckpt['optimizer_state'])

        if 'version' in checkpoint:
            print('==> Checkpoint trained from version: %s' % checkpoint['version'])
        logger.info('==> Done')

        return it, epoch

