#!/usr/bin/env python
# encoding: utf-8
'''
@author: Xu Yan
@file: basic_block.py
@time: 2021/12/16 20:34
'''
import torch
import spconv.pytorch as spconv
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.resnet import resnet34
from utils.lovasz_loss import lovasz_softmax
from mmseg.models.backbones import DINOv2, SwinTransformer
from third_party.SparseTransformer import sptr

class SparseAttenBlock(spconv.SparseModule):
    def __init__(self, in_channels, out_channels, indice_key):
        super(SparseAttenBlock, self).__init__()
        head_dim = 16
        num_heads = in_channels // head_dim
        self.atten = sptr.VarLengthMultiheadSA(
        in_channels, 
        num_heads, 
        indice_key='sptr_0', 
        window_size=6, 
        shift_win=False,
        )
        self.basicblock = SparseBasicBlock(in_channels, out_channels, indice_key)

    def forward(self, x):
        x = self.basicblock(x)
        feats, indices, spatial_shape = x.features, x.indices, x.spatial_shape
        # feats: [N, C], indices: [N, 4] with batch indices in the 0-th column
        input_tensor = sptr.SparseTrTensor(feats, indices, spatial_shape=spatial_shape, batch_size=None)
        output_tensor = self.atten(input_tensor)

        # Extract features from output tensor
        output_feats = output_tensor.query_feats
        return x.replace_feature(output_feats)

class SparseBasicBlock(spconv.SparseModule):
    def __init__(self, in_channels, out_channels, indice_key):
        super(SparseBasicBlock, self).__init__()
        self.layers_in = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, out_channels, 1, indice_key=indice_key, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        self.layers = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, out_channels, 3, indice_key=indice_key, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.1),
            spconv.SubMConv3d(out_channels, out_channels, 3, indice_key=indice_key, bias=False),
            nn.BatchNorm1d(out_channels),
        )

    def forward(self, x):
        identity = self.layers_in(x)
        output = self.layers(x)
        return output.replace_feature(F.leaky_relu(output.features + identity.features, 0.1))

class ResNet_Encoder(nn.Module):
    def __init__(self, backbone="resnet34", pretrained=True, config=None, freeze=True):
        super(ResNet_Encoder, self).__init__()
        if backbone == "resnet34":
            net = resnet34(pretrained)
        else:
            raise NotImplementedError("invalid backbone: {}".format(backbone))
        self.hiden_size = config['model_params']['hiden_size']
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.conv1.weight.data = net.conv1.weight.data
        self.bn1 = net.bn1
        self.relu = net.relu
        self.maxpool = net.maxpool
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

        if freeze:
            self._freeze()
    
    def _freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        conv1_out = self.relu(self.bn1(self.conv1(x)))
        layer1_out = self.layer1(self.maxpool(conv1_out))
        layer2_out = self.layer2(layer1_out)
        layer3_out = self.layer3(layer2_out)
        layer4_out = self.layer4(layer3_out)
        return layer1_out, layer2_out, layer3_out, layer4_out

class SwimT_Model(nn.Module):
    def __init__(self, backbone="resnet34", pretrained=True, config=None, freeze=False):
        super(SwimT_Model, self).__init__()
        swim_t_config = {'EMBED_DIM': 96,
                        'DEPTHS': [ 2, 2, 6, 2 ],
                        'NUM_HEADS': [ 3, 6, 12, 24 ],
                        'WINDOW_SIZE': 7,}
        swim_l_config = {'EMBED_DIM': 192,
                        'DEPTHS': [ 2, 2, 18, 2 ],
                        'NUM_HEADS': [ 6, 12, 24, 48 ],
                        'WINDOW_SIZE': 12,}
        self.hiden_size = config['model_params']['hiden_size']
        
        self.encoder = SwinTransformer(embed_dims=swim_t_config['EMBED_DIM'],
                                       window_size=swim_t_config['WINDOW_SIZE'],
                                       depths=swim_t_config['DEPTHS'],
                                       num_heads=swim_t_config['NUM_HEADS'],
                                       pretrained='/home/yanqiao/2DPASS/pretrained/swin_tiny_patch4_window7_224.pth')
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
        # Decoder
        self.deconv_layer1 = nn.Sequential(
            nn.Conv2d(96, self.hiden_size, kernel_size=7, stride=1, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.hiden_size, self.hiden_size, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.UpsamplingNearest2d(scale_factor=2),
        ) # [B, 96, 80, 120] -> [B, 64, 80, 120] -> [B, 64, 160, 240] -> [B, 64, 320, 480]
        self.deconv_layer2 = nn.Sequential(
            nn.Conv2d(192, self.hiden_size, kernel_size=7, stride=1, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.hiden_size, self.hiden_size, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.UpsamplingNearest2d(scale_factor=4),
        ) # [B, 192, 40, 60] -> [B, 64, 40, 60] -> [B, 64, 80, 120] -> [B, 64, 320, 480]
        self.deconv_layer3 = nn.Sequential(
            nn.Conv2d(384, self.hiden_size, kernel_size=7, stride=1, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.hiden_size, self.hiden_size, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.hiden_size, self.hiden_size, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.UpsamplingNearest2d(scale_factor=4),
        ) 
        self.deconv_layer4 = nn.Sequential(
            nn.Conv2d(768, self.hiden_size, kernel_size=7, stride=1, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.hiden_size, self.hiden_size, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.hiden_size, self.hiden_size, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.hiden_size, self.hiden_size, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.UpsamplingNearest2d(scale_factor=4),
        ) # [B, 512, 20, 30] -> [B, 64, 20, 30] -> [B, 64, 40, 60] -> [B, 64, 80, 120] -> [B, 64, 320, 480]
    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        if h % 16 != 0 or w % 16 != 0:
            assert False, "invalid input size: {}".format(x.shape)
        # Encoder: Swim-T for Image feat extraction
        layer1_out, layer2_out, layer3_out, layer4_out = self.encoder(x)
        # swim encoder feature shape: 
        # [B, 96, 80, 120]
        # [B, 192, 40, 60]
        # [B, 384, 20, 30]
        # [B, 768, 10, 15]
        layer1_out = self.deconv_layer1(layer1_out)
        layer2_out = self.deconv_layer2(layer2_out)
        layer3_out = self.deconv_layer3(layer3_out)
        layer4_out = self.deconv_layer4(layer4_out)
        return layer1_out, layer2_out, layer3_out, layer4_out


class ResNet_Model(nn.Module):
    def __init__(self, backbone="resnet34", pretrained=True, config=None, freeze=False):
        super(ResNet_Model, self).__init__()

        self.encoder = ResNet_Encoder(backbone, pretrained, config, freeze)
        self.hiden_size = config['model_params']['hiden_size']

        # Decoder
        self.deconv_layer1 = nn.Sequential(
            nn.Conv2d(64, self.hiden_size, kernel_size=7, stride=1, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.UpsamplingNearest2d(scale_factor=2),
        )
        self.deconv_layer2 = nn.Sequential(
            nn.Conv2d(128, self.hiden_size, kernel_size=7, stride=1, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.UpsamplingNearest2d(scale_factor=4),
        )
        self.deconv_layer3 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, self.hiden_size, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.UpsamplingNearest2d(scale_factor=4),
        )
        self.deconv_layer4 = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, self.hiden_size, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.UpsamplingNearest2d(scale_factor=4),
        )

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        if h % 16 != 0 or w % 16 != 0:
            assert False, "invalid input size: {}".format(x.shape)
        # Encoder
        layer1_out, layer2_out, layer3_out, layer4_out = self.encoder(x)
        # Deconv
        layer1_out = self.deconv_layer1(layer1_out)
        layer2_out = self.deconv_layer2(layer2_out)
        layer3_out = self.deconv_layer3(layer3_out)
        layer4_out = self.deconv_layer4(layer4_out)
        return layer1_out, layer2_out, layer3_out, layer4_out
    

class DepthResNetFCN(nn.Module):
    def __init__(self, depth_backbone, backbone="resnet34", pretrained=True, config=None):
        super(DepthResNetFCN, self).__init__()

        # self.image_bacbone = ResNet_Model(backbone=backbone, pretrained=pretrained, config=config, freeze=True)
        self.image_bacbone = SwimT_Model(backbone=backbone, pretrained=pretrained, config=config, freeze=True)
        self.depth_backbone = depth_backbone

    def forward(self, data_dict):       
        # color image backbone
        x = data_dict['img']
        depth_x = data_dict['depth_img']
        h, w = x.shape[2], x.shape[3]
        if h % 16 != 0 or w % 16 != 0:
            assert False, "invalid input size: {}".format(x.shape)
        layer1_out,layer2_out,layer3_out,layer4_out = self.image_bacbone(x)
        depth_layer1_out,depth_layer2_out,depth_layer3_out,depth_layer4_out = self.depth_backbone(depth_x)

        data_dict['img_scale2'] = layer1_out + depth_layer1_out
        data_dict['img_scale4'] = layer2_out + depth_layer2_out
        data_dict['img_scale8'] = layer3_out + depth_layer3_out
        data_dict['img_scale16'] = layer4_out + depth_layer4_out

        process_keys = [k for k in data_dict.keys() if k.find('img_scale') != -1]
        img_indices = data_dict['img_indices']

        temp = {k: [] for k in process_keys}

        for i in range(x.shape[0]):
            for k in process_keys:
                temp[k].append(data_dict[k].permute(0, 2, 3, 1)[i][img_indices[i][:, 0], img_indices[i][:, 1]])

        for k in process_keys:
            data_dict[k] = torch.cat(temp[k], 0)

        return data_dict

class ResNetFCN(nn.Module):
    def __init__(self, backbone="resnet34", pretrained=True, config=None):
        super(ResNetFCN, self).__init__()

        if backbone == "resnet34":
            net = resnet34(pretrained)
        else:
            raise NotImplementedError("invalid backbone: {}".format(backbone))
        self.hiden_size = config['model_params']['hiden_size']
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.conv1.weight.data = net.conv1.weight.data
        self.bn1 = net.bn1
        self.relu = net.relu
        self.maxpool = net.maxpool
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

        # Decoder
        self.deconv_layer1 = nn.Sequential(
            nn.Conv2d(64, self.hiden_size, kernel_size=7, stride=1, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.UpsamplingNearest2d(scale_factor=2),
        )
        self.deconv_layer2 = nn.Sequential(
            nn.Conv2d(128, self.hiden_size, kernel_size=7, stride=1, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.UpsamplingNearest2d(scale_factor=4),
        )
        self.deconv_layer3 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, self.hiden_size, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.UpsamplingNearest2d(scale_factor=4),
        )
        self.deconv_layer4 = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, self.hiden_size, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.UpsamplingNearest2d(scale_factor=4),
        )

    def forward(self, data_dict):
        x = data_dict['img']
        h, w = x.shape[2], x.shape[3]
        if h % 16 != 0 or w % 16 != 0:
            assert False, "invalid input size: {}".format(x.shape)

        # Encoder
        conv1_out = self.relu(self.bn1(self.conv1(x)))
        layer1_out = self.layer1(self.maxpool(conv1_out))
        layer2_out = self.layer2(layer1_out)
        layer3_out = self.layer3(layer2_out)
        layer4_out = self.layer4(layer3_out)

        # Deconv
        layer1_out = self.deconv_layer1(layer1_out)
        layer2_out = self.deconv_layer2(layer2_out)
        layer3_out = self.deconv_layer3(layer3_out)
        layer4_out = self.deconv_layer4(layer4_out)

        data_dict['img_scale2'] = layer1_out
        data_dict['img_scale4'] = layer2_out
        data_dict['img_scale8'] = layer3_out
        data_dict['img_scale16'] = layer4_out

        process_keys = [k for k in data_dict.keys() if k.find('img_scale') != -1]
        img_indices = data_dict['img_indices']

        temp = {k: [] for k in process_keys}

        for i in range(x.shape[0]):
            for k in process_keys:
                temp[k].append(data_dict[k].permute(0, 2, 3, 1)[i][img_indices[i][:, 0], img_indices[i][:, 1]])

        for k in process_keys:
            data_dict[k] = torch.cat(temp[k], 0)

        return data_dict



# class Decoder(nn.Module):
#     def __init__(self):
#         super(Decoder, self).__init__()
#         # Decoder
#         self.deconv_layer1 = nn.Sequential(
#             nn.Conv2d(96, 64, kernel_size=7, stride=1, padding=3, bias=False),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
#             nn.ReLU(inplace=True),
#             nn.UpsamplingNearest2d(scale_factor=2),
#         ) # [B, 96, 80, 120] -> [B, 64, 80, 120] -> [B, 64, 160, 240] -> [B, 64, 320, 480]
#         self.deconv_layer2 = nn.Sequential(
#             nn.Conv2d(192, 64, kernel_size=7, stride=1, padding=3, bias=False),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
#             nn.ReLU(inplace=True),
#             nn.UpsamplingNearest2d(scale_factor=4),
#         ) # [B, 192, 40, 60] -> [B, 64, 40, 60] -> [B, 64, 80, 120] -> [B, 64, 320, 480]
#         self.deconv_layer3 = nn.Sequential(
#             nn.Conv2d(384, 64, kernel_size=7, stride=1, padding=3, bias=False),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
#             nn.ReLU(inplace=True),
#             nn.UpsamplingNearest2d(scale_factor=4),
#         ) 
#         self.deconv_layer4 = nn.Sequential(
#             nn.Conv2d(768, 64, kernel_size=7, stride=1, padding=3, bias=False),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
#             nn.ReLU(inplace=True),
#             nn.UpsamplingNearest2d(scale_factor=4),
#         ) # [B, 512, 20, 30] -> [B, 64, 20, 30] -> [B, 64, 40, 60] -> [B, 64, 80, 120] -> [B, 64, 320, 480]
#     def forward(self, feats):
#         # [B, 96, 80, 120]
#         # [B, 192, 40, 60]
#         # [B, 384, 20, 30]
#         # [B, 768, 10, 15]
#         layer1_out = self.deconv_layer1(feats[0])
#         layer2_out = self.deconv_layer2(feats[1])
#         layer3_out = self.deconv_layer3(feats[2])
#         layer4_out = self.deconv_layer4(feats[3])
#         return layer1_out,layer2_out, layer3_out, layer4_out

# class DepthResNetFCN(nn.Module):
#     def __init__(self, backbone="resnet34", pretrained=True, config=None):
#         super(DepthResNetFCN, self).__init__()

#         if backbone == "resnet34":
#             net = resnet34(pretrained)
#         else:
#             raise NotImplementedError("invalid backbone: {}".format(backbone))
#         self.hiden_size = config['model_params']['hiden_size']
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
#         self.conv1.weight.data = net.conv1.weight.data
#         self.bn1 = net.bn1
#         self.relu = net.relu
#         self.maxpool = net.maxpool
#         self.layer1 = net.layer1
#         self.layer2 = net.layer2
#         self.layer3 = net.layer3
#         self.layer4 = net.layer4

#         swim_t_config = {'EMBED_DIM': 96,
#                         'DEPTHS': [ 2, 2, 6, 2 ],
#                         'NUM_HEADS': [ 3, 6, 12, 24 ],
#                         'WINDOW_SIZE': 7,}
#         swim_l_config = {'EMBED_DIM': 192,
#                         'DEPTHS': [ 2, 2, 18, 2 ],
#                         'NUM_HEADS': [ 6, 12, 24, 48 ],
#                         'WINDOW_SIZE': 12,}
        
#         self.encoder = SwinTransformer(embed_dims=swim_t_config['EMBED_DIM'],
#                                        window_size=swim_t_config['WINDOW_SIZE'],
#                                        depths=swim_t_config['DEPTHS'],
#                                        num_heads=swim_t_config['NUM_HEADS'],
#                                        pretrained='/home/yanqiao/2DPASS/pretrained/swin_tiny_patch4_window7_224.pth')
#         self.decoder = Decoder()
#         for param in self.encoder.parameters():
#             param.requires_grad = False
#         # Decoder
#         self.deconv_layer1 = nn.Sequential(
#             nn.Conv2d(64, self.hiden_size, kernel_size=7, stride=1, padding=3, bias=False),
#             nn.ReLU(inplace=True),
#             nn.UpsamplingNearest2d(scale_factor=2),
#         ) # [B, 64, 160, 240] -> [B, 64, 160, 240] -> [B, 64, 320, 480]
#         self.deconv_layer2 = nn.Sequential(
#             nn.Conv2d(128, self.hiden_size, kernel_size=7, stride=1, padding=3, bias=False),
#             nn.ReLU(inplace=True),
#             nn.UpsamplingNearest2d(scale_factor=4),
#         ) # [B, 128, 80, 120] -> [B, 64, 80, 120] -> [B, 64, 320, 480]
#         self.deconv_layer3 = nn.Sequential(
#             nn.Conv2d(256, 64, kernel_size=7, stride=1, padding=3, bias=False),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(64, self.hiden_size, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
#             nn.ReLU(inplace=True),
#             nn.UpsamplingNearest2d(scale_factor=4),
#         ) # [B, 256, 40, 60] -> [B, 64, 40, 60] -> [B, 64, 80, 120] -> [B, 64, 320, 480]
#         self.deconv_layer4 = nn.Sequential(
#             nn.Conv2d(512, 64, kernel_size=7, stride=1, padding=3, bias=False),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(64, self.hiden_size, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
#             nn.ReLU(inplace=True),
#             nn.UpsamplingNearest2d(scale_factor=4),
#         ) # [B, 512, 20, 30] -> [B, 64, 20, 30] -> [B, 64, 40, 60] -> [B, 64, 80, 120] -> [B, 64, 320, 480]

#     def forward(self, data_dict):
#         x = data_dict['img']
#         depth_x = data_dict['depth_img']
#         h, w = x.shape[2], x.shape[3]
#         if h % 16 != 0 or w % 16 != 0:
#             assert False, "invalid input size: {}".format(x.shape)

#         # Encoder: ResNet34 for depth feat extraction
#         depth_conv1_out = self.relu(self.bn1(self.conv1(depth_x)))
#         depth_layer1_out = self.layer1(self.maxpool(depth_conv1_out))
#         depth_layer2_out = self.layer2(depth_layer1_out)
#         depth_layer3_out = self.layer3(depth_layer2_out)
#         depth_layer4_out = self.layer4(depth_layer3_out)
#         #  # feature shape
#         ## conv1_out:  torch.Size([B, 64, 320, 480])
#         ## layer1_out:  torch.Size([B, 64, 160, 240])
#         ## layer2_out:  torch.Size([B, 128, 80, 120])
#         ## layer3_out:  torch.Size([B, 256, 40, 60])
#         ## layer4_out:  torch.Size([B, 512, 20, 30])
#         # Deconv
#         depth_layer1_out = self.deconv_layer1(depth_layer1_out)
#         depth_layer2_out = self.deconv_layer2(depth_layer2_out)
#         depth_layer3_out = self.deconv_layer3(depth_layer3_out)
#         depth_layer4_out = self.deconv_layer4(depth_layer4_out)

#         # Encoder: Swim-T for Image feat extraction
#         feats = self.encoder(depth_x)
#         # swim encoder feature shape: 
#         # [B, 96, 80, 120]
#         # [B, 192, 40, 60]
#         # [B, 384, 20, 30]
#         # [B, 768, 10, 15]
#         layer1_out,layer2_out, layer3_out, layer4_out = self.decoder(feats)

        

#         data_dict['img_scale2'] = layer1_out + depth_layer1_out # [B, 64, 320, 480]
#         data_dict['img_scale4'] = layer2_out + depth_layer2_out # [B, 64, 320, 480]
#         data_dict['img_scale8'] = layer3_out + depth_layer3_out # [B, 64, 320, 480]
#         data_dict['img_scale16'] = layer4_out + depth_layer4_out # [B, 64, 320, 480]

#         process_keys = [k for k in data_dict.keys() if k.find('img_scale') != -1]
#         img_indices = data_dict['img_indices']

#         temp = {k: [] for k in process_keys}

#         for i in range(x.shape[0]):
#             for k in process_keys:
#                 temp[k].append(data_dict[k].permute(0, 2, 3, 1)[i][img_indices[i][:, 0], img_indices[i][:, 1]])

#         for k in process_keys:
#             data_dict[k] = torch.cat(temp[k], 0)

#         return data_dict


class Lovasz_loss(nn.Module):
    def __init__(self, ignore=None):
        super(Lovasz_loss, self).__init__()
        self.ignore = ignore

    def forward(self, probas, labels):
        return lovasz_softmax(probas, labels, ignore=self.ignore)