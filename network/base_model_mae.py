#!/usr/bin/env python
# encoding: utf-8
'''
@author: Xu Yan
@file: base_model.py
@time: 2021/12/7 22:39
'''
import os
import torch
import yaml
import json
import numpy as np
import pytorch_lightning as pl
import cv2
from datetime import datetime
from pytorch_lightning.metrics import Accuracy
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
from utils.metric_util import IoU
from utils.schedulers import cosine_schedule_with_warmup


class LightningBaseModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # self.train_acc = Accuracy()
        # self.val_acc = Accuracy(compute_on_step=False)
        # self.val_iou = IoU(self.args['dataset_params'], compute_on_step=False)

        # if self.args['submit_to_server']:
        #     self.submit_dir = os.path.dirname(self.args['checkpoint']) + '/submit_' + datetime.now().strftime(
        #         '%Y_%m_%d')
        #     with open(self.args['dataset_params']['label_mapping'], 'r') as stream:
        #         self.mapfile = yaml.safe_load(stream)

        # self.ignore_label = self.args['dataset_params']['ignore_label']

    def configure_optimizers(self):
        if self.args['train_params']['optimizer'] == 'Adam':
            optimizer = torch.optim.AdamW(self.parameters(),
                                         lr=self.args['train_params']["learning_rate"])
        elif self.args['train_params']['optimizer'] == 'SGD':
            optimizer = torch.optim.SGD(self.parameters(),
                                        lr=self.args['train_params']["learning_rate"],
                                        momentum=self.args['train_params']["momentum"],
                                        weight_decay=self.args['train_params']["weight_decay"],
                                        nesterov=self.args['train_params']["nesterov"])
        else:
            raise NotImplementedError

        if self.args['train_params']["lr_scheduler"] == 'StepLR':
            lr_scheduler = StepLR(
                optimizer,
                step_size=self.args['train_params']["decay_step"],
                gamma=self.args['train_params']["decay_rate"]
            )
        elif self.args['train_params']["lr_scheduler"] == 'ReduceLROnPlateau':
            lr_scheduler = ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=self.args['train_params']["decay_rate"],
                patience=self.args['train_params']["decay_step"],
                verbose=True
            )
        elif self.args['train_params']["lr_scheduler"] == 'CosineAnnealingLR':
            lr_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.args['train_params']['max_num_epochs'] - 4,
                eta_min=1e-5,
            )
        elif self.args['train_params']["lr_scheduler"] == 'CosineAnnealingWarmRestarts':
            from functools import partial
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=partial(
                    cosine_schedule_with_warmup,
                    num_epochs=self.args['train_params']['max_num_epochs'],
                    batch_size=self.args['dataset_params']['train_data_loader']['batch_size'],
                    dataset_size=self.args['dataset_params']['training_size'],
                    num_gpu=len(self.args.gpu)
                ),
            )
        else:
            raise NotImplementedError

        scheduler = {
            'scheduler': lr_scheduler,
            'interval': 'step' if self.args['train_params']["lr_scheduler"] == 'CosineAnnealingWarmRestarts' else 'epoch',
            'frequency': 1
        }

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': self.args.monitor,
        }

    def forward(self, data):
        pass

    def training_step(self, data_dict, batch_idx):
        data_dict = self.forward(data_dict)
        
        # pred_img = data_dict['pred_img']
        # target_img = data_dict['img']

        # pred_img = cv2.normalize(pred_img.detach().cpu().numpy().astype(np.float32), None, 0, 255, cv2.NORM_MINMAX)
        # target_img = cv2.normalize(target_img.detach().cpu().numpy(), None, 0, 255, cv2.NORM_MINMAX)

        # pred_img = torch.from_numpy(pred_img).to(torch.uint8)
        # target_img = torch.from_numpy(target_img).to(torch.uint8)
        
        # self.train_acc(pred_img, target_img)
        return data_dict['loss']


    def validation_step(self, data_dict, batch_idx):
        data_dict = self.forward(data_dict)

        # pred_img = data_dict['pred_img']
        # target_img = data_dict['img']
        # # from IPython import embed; embed()
        # pred_img = cv2.normalize(pred_img.detach().cpu().numpy().astype(np.float32), None, 0, 255, cv2.NORM_MINMAX)
        # target_img = cv2.normalize(target_img.detach().cpu().numpy(), None, 0, 255, cv2.NORM_MINMAX)

        # pred_img = torch.from_numpy(pred_img).to(torch.uint8)
        # target_img = torch.from_numpy(target_img).to(torch.uint8)

        # self.val_acc(pred_img, target_img)
        self.log('val/acc', data_dict['loss'], on_epoch=True)

        return data_dict['loss']

    def test_step(self, data_dict, batch_idx):
        
        data_dict = self.forward(data_dict)
        
        return data_dict['loss']

    def validation_epoch_end(self, outputs):
        return

    def test_epoch_end(self, outputs):
        return 

    def on_after_backward(self) -> None:
        """
        Skipping updates in case of unstable gradients
        https://github.com/Lightning-AI/lightning/issues/4956
        """
        valid_gradients = True
        for name, param in self.named_parameters():
            if param.grad is not None:
                valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
                if not valid_gradients:
                    break
        if not valid_gradients:
            print(f'detected inf or nan values in gradients. not updating model parameters')
            self.zero_grad()