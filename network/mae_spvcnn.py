import os

import torch
import torch.nn as nn
import numpy as np

from pcdet.utils.spconv_utils import find_all_spconv_keys
from network.model_mae import mae_vit_base_patch16, mae_vit_large_patch16, mae_vit_huge_patch14, mae_vit_large_patch8, MaskedAutoencoderViT
from torch.autograd import Variable
from einops.layers.torch import Rearrange


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
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        self.sample_points = model_cfg['sample_points']
        self.input_dims = 4
        self.hiden_size = 64
        self.scale_list = [2, 4, 8, 16]
        self.num_scales = len(self.scale_list)
        min_volume_space = [-50, -50, -4]
        max_volume_space = [50, 50, 2]

        self.img_mask_ratio =  0.75
        self.img_size = (256, 1024)
        
        self.scale_factor = (16, 16)


        self.pc_encoder = SPVCNN(hidden_size=self.hiden_size, sample_points=self.sample_points)
        
        self.image_encoder = MaskedAutoencoderViT(
            patch_size=16, embed_dim=model_cfg['embed_dim'], depth=model_cfg['depth'], num_heads=model_cfg['num_heads'],
            decoder_embed_dim=model_cfg['decoder_embed_dim'], decoder_depth=model_cfg['decoder_depth'], decoder_num_heads=model_cfg['decoder_num_heads'],
            mlp_ratio=4, norm_layer=nn.LayerNorm, in_chans=3, img_with_size=self.img_size, out_chans=3, with_patch_2d=False, norm_pix_loss=True
        )
        
        self.embed_dim = model_cfg['decoder_embed_dim']
        self.decoder_embed_dim = model_cfg['decoder_embed_dim']

        self.maxpool = nn.MaxPool2d(kernel_size=16, stride=16)
        self.pts_decode = nn.Linear(self.hiden_size * 4, self.decoder_embed_dim, bias=True)

        # self.img_conv = nn.Linear(self.embed_dim, self.hiden_size, bias=True)
        self.img_conv = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.embed_dim * self.scale_factor[0] * self.scale_factor[1], kernel_size=(1, 1)),
            Rearrange('b (c s0 s1) h w -> b c (h s0) (w s1)', s0=self.scale_factor[0], s1=self.scale_factor[1]),
            nn.Conv2d(self.embed_dim, self.hiden_size, kernel_size=(1, 1)),
        )

        self.pc_decoder = SPVCNN_Decoder(hidden_size=self.hiden_size)

        print('succesfully build SPVCNN_MAE model')


    @property
    def mode(self):
        return 'TRAIN' if self.training else 'TEST'

    def update_global_step(self):
        self.global_step += 1
    
    def forward(self, batch_dict):
        return batch_dict


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

