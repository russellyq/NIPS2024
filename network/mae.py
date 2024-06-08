import os

import torch
import torch.nn as nn
import numpy as np

from pcdet.utils.spconv_utils import find_all_spconv_keys
from network.model_mae import mae_vit_base_patch16, mae_vit_large_patch16, mae_vit_huge_patch14, mae_vit_large_patch8, MaskedAutoencoderViT
from torch.autograd import Variable

class MAE(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.range_mask_ratio, self.img_mask_ratio = 0.75, 0.75
        self.img_size = (256, 1024)
        self.range_img_size = (64, 1024)

        self.image_encoder = mae_vit_large_patch16(in_chans=3, img_with_size=self.img_size, out_chans=3, with_patch_2d=False)
        self.range_encoder = mae_vit_large_patch16(in_chans=5, img_with_size=self.range_img_size, out_chans=5, with_patch_2d=(8, 8))

        self.img_patch_size = (self.image_encoder.patch_embed.patch_size[0], self.image_encoder.patch_embed.patch_size[1])
        self.range_patch_size = (self.range_encoder.patch_embed.patch_size[0], self.range_encoder.patch_embed.patch_size[1])



    def forward_patchfy_unpatchfy_img_range(self, img_latent_full, img_mask_full, img_ids_restore_full,
                                            range_latent_full, range_mask_full, range_ids_restore_full):
        
        B = img_latent_full.shape[0]
        # image embed tokens
        img_latent_full = self.image_encoder.decoder_embed(img_latent_full) # (B, L1, D=512)

        # append mask tokens to sequence
        img_mask_full_tokens = self.image_encoder.mask_token.repeat(img_latent_full.shape[0], img_ids_restore_full.shape[1] + 1 - img_latent_full.shape[1], 1) # (B, L2+1-L1=1024-256=768, D)
        x_ = torch.cat([img_latent_full[:, 1:, :], img_mask_full_tokens], dim=1)  # no cls token # (B, L2, D)
        x_ = torch.gather(x_, dim=1, index=img_ids_restore_full.unsqueeze(-1).repeat(1, 1, img_latent_full.shape[2]))  # unshuffle # (B, L2, D)
        
        img_latent_full = torch.cat([img_latent_full[:, :1, :], x_], dim=1)  # append cls token # (B, L2+1, D)

        # add pos embed
        img_latent_full = img_latent_full + self.image_encoder.decoder_pos_embed # (B, L2+1, D)
    
        # range embed tokens
        range_latent_full = self.range_encoder.decoder_embed(range_latent_full) # (B, L1, D=512)

        # append mask tokens to sequence
        range_mask_full_tokens = self.range_encoder.mask_token.repeat(range_latent_full.shape[0], range_ids_restore_full.shape[1] + 1 - range_latent_full.shape[1], 1) # (B, L2+1-L1=1024-256=768, D)
        x_ = torch.cat([range_latent_full[:, 1:, :], range_mask_full_tokens], dim=1)  # no cls token # (B, L2, D)
        x_ = torch.gather(x_, dim=1, index=range_ids_restore_full.unsqueeze(-1).repeat(1, 1, range_latent_full.shape[2]))  # unshuffle # (B, L2, D)
        
        range_latent_full = torch.cat([range_latent_full[:, :1, :], x_], dim=1)  # append cls token # (B, L2+1, D)

        # add pos embed
        range_latent_full = range_latent_full + self.range_encoder.decoder_pos_embed # (B, L2+1, D)
        
        h_img, w_img = 16, 64 # (256, 1024) / (16, 16) = (64, 1024) / (4, 16)
        img_mask_full_tokens_cls_unpatch = img_latent_full[:, 1:, :].reshape(B, h_img, w_img, img_latent_full.shape[-1])
        
        h_img, w_img = self.range_img_size[0] // self.range_patch_size[0], self.range_img_size[1] // self.range_patch_size[1]
        range_mask_full_tokens_cls_unpatch = range_latent_full[:, 1:, :].reshape(B, h_img, w_img, range_latent_full.shape[-1])

        return img_latent_full[:,:1, :], img_mask_full_tokens_cls_unpatch, range_latent_full[:, :1, :], range_mask_full_tokens_cls_unpatch
    
    def forward(self, batch_dict):
        # raw image & range image shape: (B, 3, 376, 1241)
        laser_range_in = batch_dict['laser_range_in'].permute(0,3,1,2).contiguous()
        laser_x = batch_dict['laser_x']
        laser_y = batch_dict['laser_y']
        raw_images = batch_dict['img']
        # B = batch_dict['batch_size']
        Batch_size, _, H_raw, W_raw = raw_images.size() # (256, 1024)

        images = torch.nn.functional.interpolate(raw_images, size=self.img_size, mode='bilinear')

        # color image encoding
        img_latent, img_mask, img_ids_restore = self.image_encoder.forward_encoder(images, self.img_mask_ratio)
        img_latent_full, img_mask_full, img_ids_restore_full = self.image_encoder.forward_encoder(images, 0)
        # img_latent: (B, L=H*W / 16 / 16 * (1-mask_ratio) + 1=256+1=257, C=1024)
        # img_mask: (B, L=H*W/16/16=1024)   # 0 is keep, 1 is remove
        # img_ids_restore: (B, L=H*W/16/16=1024)

        range_latent, range_mask, range_ids_restore = self.range_encoder.forward_encoder(laser_range_in, self.range_mask_ratio)
        range_latent_full, range_mask_full, range_ids_restore_full = self.range_encoder.forward_encoder(laser_range_in, 0)

        img_latent_full_cls, img_mask_full_tokens_cls_unpatch, range_latent_full_cls, range_mask_full_tokens_cls_unpatch = self.forward_patchfy_unpatchfy_img_range(img_latent_full, img_mask_full, img_ids_restore_full,
                                                                                                                        range_latent_full, range_mask_full, range_ids_restore_full)
        # (B, h_img=256/16=64/4, w_img/1024/16=1024/16, D)
        # range image 384:640 correspond to img (24:, 40)

        # h_img, w_img, D = 16, 64, img_mask_full_tokens_cls_unpatch.shape[-1]
        D = img_mask_full_tokens_cls_unpatch.shape[-1]
        h_img, w_img = self.range_img_size[0] // self.range_patch_size[0], self.range_img_size[1] // self.range_patch_size[1]
        img_token_2_range = Variable(torch.zeros((Batch_size, h_img, w_img, D), dtype=range_latent.dtype, device=range_latent.device))
        img_token_2_range[:, :, 48:80, :] = torch.nn.functional.interpolate(img_mask_full_tokens_cls_unpatch.permute(0, 3, 1, 2).contiguous(), size=(h_img, 32), mode='bilinear').permute(0, 2, 3, 1).contiguous().detach()
        img_token_2_range.requires_grad=True

        h_img, w_img = 16, 64
        # range image 384:640 correspond to img (24:, 40)
        range_token_2_img = torch.nn.functional.interpolate(range_mask_full_tokens_cls_unpatch[:, :, 48:80, :].permute(0, 3, 1, 2).contiguous(), size=(h_img, w_img), mode='bilinear').permute(0, 2, 3, 1).contiguous()

        img_token_2_range = img_token_2_range.reshape(Batch_size, -1, D)
        range_token_2_img = range_token_2_img.reshape(Batch_size, -1, D)

        img_token_2_range = torch.cat([img_latent_full_cls, img_token_2_range], 1)
        range_token_2_img = torch.cat([range_latent_full_cls, range_token_2_img], 1)

        img_pred = self.forward_decoder(img_latent, img_ids_restore, range_token_2_img,
                                        self.image_encoder.decoder_embed,
                                        self.image_encoder.mask_token,
                                        self.image_encoder.decoder_pos_embed,
                                        self.image_encoder.decoder_blocks,
                                        self.image_encoder.decoder_norm,
                                        self.image_encoder.decoder_pred)
        
        range_pred = self.forward_decoder(range_latent, range_ids_restore, img_token_2_range,
                                        self.range_encoder.decoder_embed,
                                        self.range_encoder.mask_token,
                                        self.range_encoder.decoder_pos_embed,
                                        self.range_encoder.decoder_blocks,
                                        self.range_encoder.decoder_norm,
                                        self.range_encoder.decoder_pred)   

        img_loss, img_acc = self.image_encoder.forward_loss(images, img_pred, img_mask)
        range_loss, range_acc = self.range_encoder.forward_loss(laser_range_in, range_pred, range_mask, patch_size=(4, 16))

        loss = range_loss + img_loss

        # from IPython import embed; embed()
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

            pred_dicts['img_pred'] = self.image_encoder.unpatchify(img_pred, h=16, w=64).permute(0,2,3,1).squeeze(0).detach().cpu().numpy()
            pred_dicts['img_mask'] = self.image_encoder.unpatchify(img_mask.unsqueeze(-1).repeat(1, 1, self.image_encoder.patch_embed.patch_size[0]**2 *3), h=16, w=64).permute(0,2,3,1).squeeze(0).detach().cpu().numpy()
            pred_dicts['img'] = raw_images.permute(0,2,3,1).squeeze(0).detach().cpu().numpy()
            pred_dicts['img_raw'] = images.permute(0,2,3,1).squeeze(0).detach().cpu().numpy()
            pred_dicts['im_masked'] = pred_dicts['img_raw'] * (1 - pred_dicts['img_mask'])
            pred_dicts['im_paste'] = pred_dicts['im_masked'] + pred_dicts['img_pred'] * pred_dicts['img_mask']

            pred_dicts['range_acc'] = range_acc.detach().cpu().numpy()

            pred_dicts['range_pred'] = self.range_encoder.unpatchify(range_pred, h=8, w=128, C=5).permute(0,2,3,1).squeeze(0).detach().cpu().numpy()
            pred_dicts['range_mask'] = self.range_encoder.unpatchify(range_mask.unsqueeze(-1).repeat(1, 1, self.range_encoder.patch_embed.patch_size[0]**2 *5), h=8, w=128, C=5).permute(0,2,3,1).squeeze(0).detach().cpu().numpy()
            pred_dicts['range_raw'] = laser_range_in.permute(0,2,3,1).squeeze(0).detach().cpu().numpy()
            pred_dicts['range_masked'] = pred_dicts['range_raw'] * (1 - pred_dicts['range_mask'])
            pred_dicts['range_paste'] = pred_dicts['range_masked'] + pred_dicts['range_pred'] * pred_dicts['range_mask']

            # from IPython import embed; embed()

            
            return pred_dicts, recall_dicts
        
        return batch_dict
    
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

        x = x + x_mm
        
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


    def get_training_loss(self):
        loss = 0
        disp_dict = {}
        tb_dict = {}
        return loss, tb_dict, disp_dict
    
    def post_processing(self, batch_dict):
        pred_dicts, recall_dict= {}, {}
        return pred_dicts, recall_dict


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
                print('Update weight %s: %s' % (key, str(val.shape)))

        if strict:
            self.load_state_dict(update_model_state)
        else:
            state_dict.update(update_model_state)
            self.load_state_dict(state_dict)
        return state_dict, update_model_state

    def load_params_from_file(self, filename, logger, to_cpu=False, pre_trained_path=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        print('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['model_state']
        if not pre_trained_path is None:
            pretrain_checkpoint = torch.load(pre_trained_path, map_location=loc_type)
            pretrain_model_state_disk = pretrain_checkpoint['model_state']
            model_state_disk.update(pretrain_model_state_disk)
            
        version = checkpoint.get("version", None)
        if version is not None:
            print('==> Checkpoint trained from version: %s' % version)

        state_dict, update_model_state = self._load_state_dict(model_state_disk, strict=False)

        for key in state_dict:
            if key not in update_model_state:
                print('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))

        # print('==> Done (loaded %d/%d)' % (len(update_model_state), len(state_dict)))

    def load_params_with_optimizer(self, filename, to_cpu=False, optimizer=None, logger=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        # print('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        epoch = checkpoint.get('epoch', -1)
        it = checkpoint.get('it', 0.0)

        self._load_state_dict(checkpoint['model_state'], strict=True)

        if optimizer is not None:
            if 'optimizer_state' in checkpoint and checkpoint['optimizer_state'] is not None:
                print('==> Loading optimizer parameters from checkpoint %s to %s'
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
        print('==> Done')

        return it, epoch

class MAE_Range(nn.Module):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.dataset = dataset
        self.register_buffer('global_step', torch.LongTensor(1).zero_())

        self.module_topology = [
            'vfe', 'backbone_3d', 'map_to_bev_module', 'pfe',
            'backbone_2d', 'dense_head',  'point_head', 'roi_head'
        ]
        self.range_mask_ratio, self.img_mask_ratio = 0.75, 0.75
        self.range_img_size = (64, 1024)

        self.range_encoder = mae_vit_large_patch16(in_chans=5, img_with_size=self.range_img_size, out_chans=5, with_patch_2d=(8, 8))
        self.range_patch_size = (self.range_encoder.patch_embed.patch_size[0], self.range_encoder.patch_embed.patch_size[1])
        self.img_size = (256, 1024)

    @property
    def mode(self):
        return 'TRAIN' if self.training else 'TEST'

    def update_global_step(self):
        self.global_step += 1

  
    def forward(self, batch_dict):
        # raw image & range image shape: (B, 3, 376, 1241)
        laser_range_in = batch_dict['laser_range_in'].permute(0,3,1,2).contiguous()
        laser_x = batch_dict['laser_x']
        laser_y = batch_dict['laser_y']
        raw_images = batch_dict['images']
        # B = batch_dict['batch_size']
        Batch_size, _, H_raw, W_raw = raw_images.size() # (256, 1024)

        images = torch.nn.functional.interpolate(raw_images, size=self.img_size, mode='bilinear')

        range_latent, range_mask, range_ids_restore = self.range_encoder.forward_encoder(laser_range_in, self.range_mask_ratio)

        range_pred = self.forward_decoder(range_latent, range_ids_restore,
                                        self.range_encoder.decoder_embed,
                                        self.range_encoder.mask_token,
                                        self.range_encoder.decoder_pos_embed,
                                        self.range_encoder.decoder_blocks,
                                        self.range_encoder.decoder_norm,
                                        self.range_encoder.decoder_pred)   

        img_loss = 0
        range_loss, range_acc = self.range_encoder.forward_loss(laser_range_in, range_pred, range_mask, patch_size=(8, 8))

        loss = range_loss + img_loss

        # from IPython import embed; embed()
        if self.training:
            tb_dict = {
                # 'range_roi_loss': range_roi_loss,
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

            pred_dicts['img'] = raw_images.permute(0,2,3,1).squeeze(0).detach().cpu().numpy()
            pred_dicts['img_raw'] = images.permute(0,2,3,1).squeeze(0).detach().cpu().numpy()

            pred_dicts['range_acc'] = range_acc.detach().cpu().numpy()

            pred_dicts['range_pred'] = self.range_encoder.unpatchify(range_pred, h=8, w=128, C=5).permute(0,2,3,1).squeeze(0).detach().cpu().numpy()
            pred_dicts['range_mask'] = self.range_encoder.unpatchify(range_mask.unsqueeze(-1).repeat(1, 1, self.range_encoder.patch_embed.patch_size[0]**2 *5), h=8, w=128, C=5).permute(0,2,3,1).squeeze(0).detach().cpu().numpy()
            pred_dicts['range_raw'] = laser_range_in.permute(0,2,3,1).squeeze(0).detach().cpu().numpy()
            pred_dicts['range_masked'] = pred_dicts['range_raw'] * (1 - pred_dicts['range_mask'])
            pred_dicts['range_paste'] = pred_dicts['range_masked'] + pred_dicts['range_pred'] * pred_dicts['range_mask']

            # from IPython import embed; embed()

            
            return pred_dicts, recall_dicts
        
        return batch_dict
    
    def forward_decoder(self, x, ids_restore, \
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


    def get_training_loss(self):
        loss = 0
        disp_dict = {}
        tb_dict = {}
        return loss, tb_dict, disp_dict
    
    def post_processing(self, batch_dict):
        pred_dicts, recall_dict= {}, {}
        return pred_dicts, recall_dict


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
                # print('Update weight %s: %s' % (key, str(val.shape)))

        if strict:
            self.load_state_dict(update_model_state)
        else:
            state_dict.update(update_model_state)
            self.load_state_dict(state_dict)
        return state_dict, update_model_state

    def load_params_from_file(self, filename, logger, to_cpu=False, pre_trained_path=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        print('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['model_state']
        if not pre_trained_path is None:
            pretrain_checkpoint = torch.load(pre_trained_path, map_location=loc_type)
            pretrain_model_state_disk = pretrain_checkpoint['model_state']
            model_state_disk.update(pretrain_model_state_disk)
            
        version = checkpoint.get("version", None)
        if version is not None:
            print('==> Checkpoint trained from version: %s' % version)

        state_dict, update_model_state = self._load_state_dict(model_state_disk, strict=False)

        for key in state_dict:
            if key not in update_model_state:
                print('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))

        print('==> Done (loaded %d/%d)' % (len(update_model_state), len(state_dict)))

    def load_params_with_optimizer(self, filename, to_cpu=False, optimizer=None, logger=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        print('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        epoch = checkpoint.get('epoch', -1)
        it = checkpoint.get('it', 0.0)

        self._load_state_dict(checkpoint['model_state'], strict=True)

        if optimizer is not None:
            if 'optimizer_state' in checkpoint and checkpoint['optimizer_state'] is not None:
                print('==> Loading optimizer parameters from checkpoint %s to %s'
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
        print('==> Done')

        return it, epoch

class MAE_Image(nn.Module):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.dataset = dataset
        self.register_buffer('global_step', torch.LongTensor(1).zero_())

        self.module_topology = [
            'vfe', 'backbone_3d', 'map_to_bev_module', 'pfe',
            'backbone_2d', 'dense_head',  'point_head', 'roi_head'
        ]
        self.range_mask_ratio, self.img_mask_ratio = 0.75, 0.75
        self.img_size = (256, 1024)
        self.range_img_size = (64, 1024)

        self.image_encoder = mae_vit_large_patch16(in_chans=3, img_with_size=self.img_size, out_chans=3, with_patch_2d=False)


    @property
    def mode(self):
        return 'TRAIN' if self.training else 'TEST'

    def update_global_step(self):
        self.global_step += 1


    def forward(self, batch_dict):
        # raw image & range image shape: (B, 3, 376, 1241)
        raw_images = batch_dict['images']
        # B = batch_dict['batch_size']
        Batch_size, _, H_raw, W_raw = raw_images.size() # (256, 1024)

        images = torch.nn.functional.interpolate(raw_images, size=self.img_size, mode='bilinear')

        # color image encoding
        img_latent, img_mask, img_ids_restore = self.image_encoder.forward_encoder(images, self.img_mask_ratio)
        # img_latent: (B, L=H*W / 16 / 16 * (1-mask_ratio) + 1=256+1=257, C=1024)
        # img_mask: (B, L=H*W/16/16=1024)   # 0 is keep, 1 is remove
        # img_ids_restore: (B, L=H*W/16/16=1024)


        img_pred = self.forward_decoder(img_latent, img_ids_restore,
                                        self.image_encoder.decoder_embed,
                                        self.image_encoder.mask_token,
                                        self.image_encoder.decoder_pos_embed,
                                        self.image_encoder.decoder_blocks,
                                        self.image_encoder.decoder_norm,
                                        self.image_encoder.decoder_pred)
        

        img_loss, img_acc = self.image_encoder.forward_loss(images, img_pred, img_mask)
        range_loss = 0
        loss = range_loss + img_loss

        # from IPython import embed; embed()
        if self.training:
            tb_dict = {
                # 'range_roi_loss': range_roi_loss,
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

            # pred_dicts['img_pred'] = self.image_encoder.unpatchify(img_pred, h=4, w=64).permute(0,2,3,1).squeeze(0).detach().cpu().numpy()
            # pred_dicts['img_mask'] = self.image_encoder.unpatchify(img_mask.unsqueeze(-1).repeat(1, 1, self.image_encoder.patch_embed.patch_size[0]**2 *3), h=4, w=64).permute(0,2,3,1).squeeze(0).detach().cpu().numpy()
            # pred_dicts['img'] = raw_images.permute(0,2,3,1).squeeze(0).detach().cpu().numpy()
            # pred_dicts['img_raw'] = images.permute(0,2,3,1).squeeze(0).detach().cpu().numpy()
            # pred_dicts['im_masked'] = pred_dicts['img_raw'] * (1 - pred_dicts['img_mask'])
            # pred_dicts['im_paste'] = pred_dicts['im_masked'] + pred_dicts['img_pred'] * pred_dicts['img_mask']

            # pred_dicts['range_roi_pred'] = self.range_encoder.unpatchify(range_pred, h=4, w=64, C=5).permute(0,2,3,1).squeeze(0).detach().cpu().numpy()
            # pred_dicts['range_roi_mask'] = self.range_encoder.unpatchify(range_mask_roi.unsqueeze(-1).repeat(1, 1, self.range_encoder.patch_embed.patch_size[0]**2 *5), h=4, w=64, C=5).permute(0,2,3,1).squeeze(0).detach().cpu().numpy()
            # pred_dicts['range_roi_raw'] = laser_range_in_roi.permute(0,2,3,1).squeeze(0).detach().cpu().numpy()
            # pred_dicts['range_roi_masked'] = pred_dicts['range_roi_raw'] * (1 - pred_dicts['range_roi_mask'])
            # pred_dicts['range_roi_paste'] = pred_dicts['range_roi_masked'] + pred_dicts['range_roi_pred'] * pred_dicts['range_roi_mask']
            
            
            # pred_dicts['range_pred'] = self.range_encoder.unpatchify(range_pred, h=4, w=64, C=5).permute(0,2,3,1).squeeze(0).detach().cpu().numpy()
            # pred_dicts['range_mask'] = self.range_encoder.unpatchify(range_mask.unsqueeze(-1).repeat(1, 1, self.range_encoder.patch_embed.patch_size[0]**2 *5), h=4, w=64, C=5).permute(0,2,3,1).squeeze(0).detach().cpu().numpy()
            # pred_dicts['range_raw'] = laser_range_in.permute(0,2,3,1).squeeze(0).detach().cpu().numpy()
            # pred_dicts['range_masked'] = pred_dicts['range_raw'] * (1 - pred_dicts['range_mask'])
            # pred_dicts['range_paste'] = pred_dicts['range_masked'] + pred_dicts['range_pred'] * pred_dicts['range_mask']

            # from IPython import embed; embed()

            
            return pred_dicts, recall_dicts
        
        return batch_dict
    
    def forward_decoder(self, x, ids_restore, \
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


    def get_training_loss(self):
        loss = 0
        disp_dict = {}
        tb_dict = {}
        return loss, tb_dict, disp_dict
    
    def post_processing(self, batch_dict):
        pred_dicts, recall_dict= {}, {}
        return pred_dicts, recall_dict


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
                # print('Update weight %s: %s' % (key, str(val.shape)))

        if strict:
            self.load_state_dict(update_model_state)
        else:
            state_dict.update(update_model_state)
            self.load_state_dict(state_dict)
        return state_dict, update_model_state

    def load_params_from_file(self, filename, logger, to_cpu=False, pre_trained_path=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        print('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['model_state']
        if not pre_trained_path is None:
            pretrain_checkpoint = torch.load(pre_trained_path, map_location=loc_type)
            pretrain_model_state_disk = pretrain_checkpoint['model_state']
            model_state_disk.update(pretrain_model_state_disk)
            
        version = checkpoint.get("version", None)
        if version is not None:
            print('==> Checkpoint trained from version: %s' % version)

        state_dict, update_model_state = self._load_state_dict(model_state_disk, strict=False)

        for key in state_dict:
            if key not in update_model_state:
                print('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))

        print('==> Done (loaded %d/%d)' % (len(update_model_state), len(state_dict)))

    def load_params_with_optimizer(self, filename, to_cpu=False, optimizer=None, logger=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        print('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        epoch = checkpoint.get('epoch', -1)
        it = checkpoint.get('it', 0.0)

        self._load_state_dict(checkpoint['model_state'], strict=True)

        if optimizer is not None:
            if 'optimizer_state' in checkpoint and checkpoint['optimizer_state'] is not None:
                print('==> Loading optimizer parameters from checkpoint %s to %s'
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
        print('==> Done')

        return it, epoch
