#!/usr/bin/env python
# encoding: utf-8
'''
@author: Xu Yan
@file: main.py
@time: 2021/12/7 22:21
'''
from PIL import Image
import os
import yaml
import torch
import datetime
import importlib
import numpy as np
import pytorch_lightning as pl

from easydict import EasyDict
from argparse import ArgumentParser
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.profiler import SimpleProfiler
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from dataloader.dataset import get_model_class, get_collate_class
from dataloader.pc_dataset import get_pc_model_class
from pytorch_lightning.callbacks import LearningRateMonitor
import time
import warnings
warnings.filterwarnings("ignore")
import cv2
from dataloader.laserscan import LaserScan

def load_yaml(file_name):
    with open(file_name, 'r') as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            config = yaml.load(f)
    return config


def parse_config():
    parser = ArgumentParser()
    # general
    parser.add_argument('--gpu', type=int, nargs='+', default=(0,1), help='specify gpu devices')
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument('--config_path', default='config/2DPASS-semantickitti.yaml')
    # training
    parser.add_argument('--log_dir', type=str, default='default', help='log location')
    parser.add_argument('--monitor', type=str, default='val/mIoU', help='the maximum metric')
    parser.add_argument('--stop_patience', type=int, default=50, help='patience for stop training')
    parser.add_argument('--save_top_k', type=int, default=1, help='save top k checkpoints, use -1 to checkpoint every epoch')
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1, help='check_val_every_n_epoch')
    parser.add_argument('--SWA', action='store_true', default=False, help='StochasticWeightAveraging')
    parser.add_argument('--baseline_only', action='store_true', default=False, help='training without 2D')
    # testing
    parser.add_argument('--test', action='store_true', default=False, help='test mode')
    parser.add_argument('--fine_tune', action='store_true', default=False, help='fine tune mode')
    parser.add_argument('--pretrain2d', action='store_true', default=False, help='use pre-trained 2d network')
    parser.add_argument('--num_vote', type=int, default=1, help='number of voting in the test')
    parser.add_argument('--submit_to_server', action='store_true', default=False, help='submit on benchmark')
    parser.add_argument('--checkpoint', type=str, default=None, help='load checkpoint')
    # debug
    parser.add_argument('--debug', default=False, action='store_true')

    args = parser.parse_args()
    config = load_yaml(args.config_path)
    config.update(vars(args))  # override the configuration using the value in args

    # voting test
    if args.test:
        config['dataset_params']['val_data_loader']['batch_size'] = args.num_vote
    if args.num_vote > 1:
        config['dataset_params']['val_data_loader']['rotate_aug'] = True
        config['dataset_params']['val_data_loader']['transform_aug'] = True
    if args.debug:
        config['dataset_params']['val_data_loader']['batch_size'] = 2
        config['dataset_params']['val_data_loader']['num_workers'] = 0

    return EasyDict(config)


if __name__ == '__main__':
    # parameters
    configs = parse_config()
    print(configs)

    pc_dataset = get_pc_model_class(configs['dataset_params']['pc_dataset_type'])
    dataset_type = get_model_class(configs['dataset_params']['dataset_type'])
    train_config = configs['dataset_params']['train_data_loader']
    val_config = configs['dataset_params']['val_data_loader']
    val_pt_dataset = pc_dataset(configs, data_path=val_config['data_path'], imageset='val')
    laser = LaserScan()

    val_dataset = dataset_type(val_pt_dataset, configs, val_config, num_vote=1)

    
        
    # # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, configs.gpu))
    num_gpu = len(configs.gpu)

    model_file = importlib.import_module('network.' + configs['model_params']['model_architecture'])
    my_model = model_file.get_model(configs)

    device = torch.device('cpu')

    my_model = my_model.load_from_checkpoint(configs.checkpoint, config=configs)

    my_model.to(device)
    my_model.eval()

    root_dir = '/media/yanqiao/Seagate Hub/semantic-kitti/dataset/sequences/08/image_2_pred/'
    os.makedirs(root_dir, exist_ok=True)
    with torch.no_grad():
        for i in range(len(val_dataset)):
            print(i)
            data_dict = val_dataset[i]

            proj_range_img = data_dict['proj_range_img']
            img = data_dict['img']

            proj_range_img_tensor = torch.from_numpy(proj_range_img).to(device).permute(2,0,1).unsqueeze(0)
            img_tensor = torch.from_numpy(img).to(device).permute(2,0,1).unsqueeze(0)

            input_dict ={
                'img': img_tensor,
                'proj_range_img': proj_range_img_tensor,
            }

            input_dict = my_model(input_dict)

            pred_img_tensor = input_dict['pred_img']

            pred_img = cv2.normalize(pred_img_tensor.squeeze(0).permute(1,2,0).detach().cpu().numpy().astype(np.float32), None, 0, 255, cv2.NORM_MINMAX)

            print('pred_img_tensor: ', pred_img_tensor.size())

            file_name = root_dir + '%06d.png' % i

            cv2.imwrite(file_name, pred_img)




    # root_dir = '/media/yanqiao/Seagate Hub/semantic-kitti/dataset/sequences/'
    # folder_list = ['08']

    # for i in folder_list:
    #     path = root_dir + str(i)
    #     image_dir = path + '/image_2'
    #     pc_dir = path + '/velodyne'


    #     image_files = os.listdir(image_dir)
    #     number = 0
    #     for image_file in image_files:
    #         t0 = time.time()
    #         number += 1
    #         image_index = image_file[:-4]
    #         image_path = image_dir + '/' + image_file
    #         pc_path = pc_dir + '/' + image_file[:-4] + '.bin'
    #         image = Image.open(image_path)
    #         pc = np.fromfile(pc_path, dtype=np.float32).reshape((-1, 4))

    #         print(f"Processing '{image_path}'...")
    #         print("Processing number: ", number)

            

