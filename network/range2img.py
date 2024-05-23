import torch
import torch_scatter
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from network.base_model_mae import LightningBaseModel

# from models_mae import mae_vit_base_patch16, mae_vit_large_patch16, mae_vit_huge_patch14
from network.models_mae import mae_vit_base_patch16, mae_vit_large_patch16, mae_vit_huge_patch14

# class get_model(pl.LightningModule):
class get_model(LightningBaseModel):
    def __init__(self, config):
        super(get_model, self).__init__(config)
        self.mae = mae_vit_large_patch16(in_chans=5, img_with_size=(256, 1024))
        
        
    def forward(self, data_dict):
        with torch.cuda.amp.autocast():
            data_dict, loss, _, _ = self.mae(data_dict)
            data_dict['loss'] = loss
        # from IPython import embed; embed()
        return data_dict
    

if __name__ == '__main__':
    range_imgs = torch.randn(12, 5, 64, 1024)
    imgs = torch.randn(12, 3, 376, 1241)
    model = get_model(config=None)
    data_dict = {
        'range_imgs': range_imgs,
        'img': imgs,
    }
    data_dict = model(data_dict)