"""
Task-specific Datasets
"""
import random
import torch
import numpy as np
import cv2
from PIL import Image
from torch.utils import data
from torchvision import transforms as T
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import view_points
from .laserscan import LaserScan
from scipy.spatial.ckdtree import cKDTree as kdtree
import random

def make_random_range_mask(patch_size_h=2, patch_size_w=8, H=64, W=2048, mask_ratio=0.75):
    t_or_f_h = []
    for i in range(H // patch_size_h):
        t_or_f_w = []
        t_or_f = []
        w_mask_len = int( ( W // patch_size_w ) * mask_ratio)
        w_unmask_len = int( ( W // patch_size_w ) * ( 1 - mask_ratio ))
        for i in range(w_mask_len):
            t_or_f.append(False)
        for i in range(w_unmask_len):
            t_or_f.append(True)
        
        random.shuffle(t_or_f) # (256)
        t_or_f = np.asarray(t_or_f) # () # (256)
        t_or_f = t_or_f.repeat(patch_size_w) # (2048)
        t_or_f = t_or_f.reshape(1, -1) # (1, 2048)
        for i in range(patch_size_h):
            t_or_f_w.append(t_or_f)
        t_or_f_w = np.concatenate(t_or_f_w, axis=0) # (2, 2048)
        
        t_or_f_h.append(t_or_f_w)
    t_or_f_h = np.concatenate(t_or_f_h, axis=0)
    return t_or_f_h

REGISTERED_DATASET_CLASSES = {}
REGISTERED_COLATE_CLASSES = {}

try:
    from torchsparse import SparseTensor
    from torchsparse.utils.collate import sparse_collate_fn
    from torchsparse.utils.quantize import sparse_quantize
except:
    print('please install torchsparse if you want to run spvcnn/minkowskinet!')


def register_dataset(cls, name=None):
    global REGISTERED_DATASET_CLASSES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_DATASET_CLASSES, f"exist class: {REGISTERED_DATASET_CLASSES}"
    REGISTERED_DATASET_CLASSES[name] = cls
    return cls


def register_collate_fn(cls, name=None):
    global REGISTERED_COLATE_CLASSES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_COLATE_CLASSES, f"exist class: {REGISTERED_COLATE_CLASSES}"
    REGISTERED_COLATE_CLASSES[name] = cls
    return cls


def get_model_class(name):
    global REGISTERED_DATASET_CLASSES
    assert name in REGISTERED_DATASET_CLASSES, f"available class: {REGISTERED_DATASET_CLASSES}"
    return REGISTERED_DATASET_CLASSES[name]


def get_collate_class(name):
    global REGISTERED_COLATE_CLASSES
    assert name in REGISTERED_COLATE_CLASSES, f"available class: {REGISTERED_COLATE_CLASSES}"
    return REGISTERED_COLATE_CLASSES[name]


@register_dataset
class point_image_dataset_semkitti(data.Dataset):
    def __init__(self, in_dataset, config, loader_config, num_vote=1, trans_std=[0.1, 0.1, 0.1], max_dropout_ratio=0.2):
        'Initialization'
        self.point_cloud_dataset = in_dataset
        self.config = config
        self.ignore_label = config['dataset_params']['ignore_label']
        self.rotate_aug = loader_config['rotate_aug']
        self.flip_aug = loader_config['flip_aug']
        self.transform = loader_config['transform_aug']
        self.scale_aug = loader_config['scale_aug']
        self.dropout = loader_config['dropout_aug']
        self.instance_aug = loader_config.get('instance_aug', False)
        self.max_volume_space = config['dataset_params']['max_volume_space']
        self.min_volume_space = config['dataset_params']['min_volume_space']
        self.num_vote = num_vote
        self.trans_std = trans_std
        self.max_dropout_ratio = max_dropout_ratio
        self.debug = config['debug']

        self.bottom_crop = config['dataset_params']['bottom_crop']
        color_jitter = config['dataset_params']['color_jitter']
        self.color_jitter = T.ColorJitter(*color_jitter) if color_jitter else None
        self.flip2d = config['dataset_params']['flip2d']
        self.image_normalizer = config['dataset_params']['image_normalizer']
        # self.laserscaner = LaserScan()

    def __len__(self):
        'Denotes the total number of samples'
        if self.debug:
            return 100 * self.num_vote
        else:
            return len(self.point_cloud_dataset)

    @staticmethod
    def select_points_in_frustum(points_2d, x1, y1, x2, y2):
        """
        Select points in a 2D frustum parametrized by x1, y1, x2, y2 in image coordinates
        :param points_2d: point cloud projected into 2D
        :param points_3d: point cloud
        :param x1: left bound
        :param y1: upper bound
        :param x2: right bound
        :param y2: lower bound
        :return: points (2D and 3D) that are in the frustum
        """
        keep_ind = (points_2d[:, 0] > x1) * \
                   (points_2d[:, 1] > y1) * \
                   (points_2d[:, 0] < x2) * \
                   (points_2d[:, 1] < y2)

        return keep_ind

    def __getitem__(self, index):
        'Generates one sample of data'
        data, root = self.point_cloud_dataset[index]

        xyz = data['xyz']
        labels = data['labels']
        instance_label = data['instance_label'].reshape(-1)
        sig = data['signal']
        origin_len = data['origin_len']

        ref_pc = xyz.copy()
        ref_labels = labels.copy()
        ref_index = np.arange(len(ref_pc))

        mask_x = np.logical_and(xyz[:, 0] > self.min_volume_space[0], xyz[:, 0] < self.max_volume_space[0])
        mask_y = np.logical_and(xyz[:, 1] > self.min_volume_space[1], xyz[:, 1] < self.max_volume_space[1])
        mask_z = np.logical_and(xyz[:, 2] > self.min_volume_space[2], xyz[:, 2] < self.max_volume_space[2])
        mask = np.logical_and(mask_x, np.logical_and(mask_y, mask_z))

        xyz = xyz[mask]
        ref_pc = ref_pc[mask]
        labels = labels[mask]
        instance_label = instance_label[mask]
        ref_index = ref_index[mask]
        sig = sig[mask]
        point_num = len(xyz)

        if self.dropout and self.point_cloud_dataset.imageset == 'train':
            dropout_ratio = np.random.random() * self.max_dropout_ratio
            drop_idx = np.where(np.random.random((xyz.shape[0])) <= dropout_ratio)[0]

            if len(drop_idx) > 0:
                xyz[drop_idx, :] = xyz[0, :]
                labels[drop_idx, :] = labels[0, :]
                sig[drop_idx, :] = sig[0, :]
                instance_label[drop_idx] = instance_label[0]
                ref_index[drop_idx] = ref_index[0]

        # load 2D data
        image = data['img']
        # depth_image = data['depth_img']
        proj_matrix = data['proj_matrix']

        # project points into image
        keep_idx = xyz[:, 0] > 0  # only keep point in front of the vehicle
        points_hcoords = np.concatenate([xyz[keep_idx], np.ones([keep_idx.sum(), 1], dtype=np.float32)], axis=1)
        img_points = (proj_matrix @ points_hcoords.T).T
        img_points = img_points[:, :2] / np.expand_dims(img_points[:, 2], axis=1)  # scale 2D points
        keep_idx_img_pts = self.select_points_in_frustum(img_points, 0, 0, *image.size)
        keep_idx[keep_idx] = keep_idx_img_pts

        # fliplr so that indexing is row, col and not col, row
        img_points = np.fliplr(img_points)
        points_img = img_points[keep_idx_img_pts]

        ### 3D Augmentation ###
        # random data augmentation by rotation
        if self.rotate_aug:
            rotate_rad = np.deg2rad(np.random.random() * 360)
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            xyz[:, :2] = np.dot(xyz[:, :2], j)

        # random data augmentation by flip x , y or x+y
        if self.flip_aug:
            flip_type = np.random.choice(4, 1)
            if flip_type == 1:
                xyz[:, 0] = -xyz[:, 0]
            elif flip_type == 2:
                xyz[:, 1] = -xyz[:, 1]
            elif flip_type == 3:
                xyz[:, :2] = -xyz[:, :2]

        if self.scale_aug:
            noise_scale = np.random.uniform(0.95, 1.05)
            xyz[:, 0] = noise_scale * xyz[:, 0]
            xyz[:, 1] = noise_scale * xyz[:, 1]

        if self.transform:
            noise_translate = np.array([np.random.normal(0, self.trans_std[0], 1),
                                        np.random.normal(0, self.trans_std[1], 1),
                                        np.random.normal(0, self.trans_std[2], 1)]).T

            xyz[:, 0:3] += noise_translate

        img_label = labels[keep_idx]
        point2img_index = np.arange(len(labels))[keep_idx]
        feat = np.concatenate((xyz, sig), axis=1)

        # processing to range image
        self.laserscaner.open_scan_with_points(xyz, sig)
        proj_range = self.laserscaner.proj_range # [H, W]
        proj_xyz = self.laserscaner.proj_xyz # [H, W, 3]
        proj_remission = self.laserscaner.proj_remission # [H, W]
        proj_idx = self.laserscaner.proj_idx # [H, W]
        proj_mask = self.laserscaner.proj_mask # [H, W]
        proj_x = self.laserscaner.proj_x
        proj_y = self.laserscaner.proj_y

        proj_idx = proj_idx[:, :, np.newaxis]
        proj_range = proj_range[:, :, np.newaxis]
        proj_remission = proj_remission[:, :, np.newaxis]
        proj_range_img = np.concatenate([proj_xyz, proj_range, proj_remission], 2)
        proj_range = np.concatenate([proj_range, proj_range, proj_range], 2)
        proj_remission = np.concatenate([proj_remission, proj_remission, proj_remission], 2)
        proj_range = cv2.normalize(proj_range, None, 0, 1, cv2.NORM_MINMAX)
        proj_remission = cv2.normalize(proj_remission, None, 0, 1, cv2.NORM_MINMAX)
        proj_xyz = cv2.normalize(proj_xyz, None, 0, 1, cv2.NORM_MINMAX)
        proj_range_img = cv2.normalize(proj_range_img, None, 0, 1, cv2.NORM_MINMAX)
        ### 2D Augmentation ###
        if self.bottom_crop:
            # self.bottom_crop is a tuple (crop_width, crop_height)
            left = int(np.random.rand() * (image.size[0] + 1 - self.bottom_crop[0]))
            right = left + self.bottom_crop[0]
            top = image.size[1] - self.bottom_crop[1]
            bottom = image.size[1]

            # update image points
            keep_idx = points_img[:, 0] >= top
            keep_idx = np.logical_and(keep_idx, points_img[:, 0] < bottom)
            keep_idx = np.logical_and(keep_idx, points_img[:, 1] >= left)
            keep_idx = np.logical_and(keep_idx, points_img[:, 1] < right)

            # crop image
            image = image.crop((left, top, right, bottom))
            # depth_image = depth_image.crop((left, top, right, bottom))
            points_img = points_img[keep_idx]
            points_img[:, 0] -= top
            points_img[:, 1] -= left

            img_label = img_label[keep_idx]
            point2img_index = point2img_index[keep_idx]

        img_indices = points_img.astype(np.int64)

        # 2D augmentation
        if self.color_jitter is not None:
            image = self.color_jitter(image)
            # depth_image = self.color_jitter(depth_image)

        # PIL to numpy
        image = np.array(image, dtype=np.float32, copy=False) / 255.
        # depth_image = np.array(depth_image, dtype=np.float32, copy=False) / 255.

        # 2D augmentation
        if np.random.rand() < self.flip2d:
            image = np.ascontiguousarray(np.fliplr(image))
            # depth_image = np.ascontiguousarray(np.fliplr(depth_image))
            img_indices[:, 1] = image.shape[1] - 1 - img_indices[:, 1]

        # normalize image
        if self.image_normalizer:
            mean, std = self.image_normalizer
            mean = np.asarray(mean, dtype=np.float32)
            std = np.asarray(std, dtype=np.float32)
            image = (image - mean) / std
            # depth_image = (depth_image - mean) / std

        data_dict = {}
        data_dict['point_feat'] = feat
        data_dict['point_label'] = labels
        data_dict['ref_xyz'] = ref_pc
        data_dict['ref_label'] = ref_labels
        data_dict['ref_index'] = ref_index
        data_dict['mask'] = mask
        data_dict['point_num'] = point_num
        data_dict['origin_len'] = origin_len
        data_dict['root'] = root

        data_dict['img'] = image
        # data_dict['depth_img'] = depth_image
        data_dict['img_indices'] = img_indices
        data_dict['img_label'] = img_label
        data_dict['point2img_index'] = point2img_index

        data_dict['proj_xyz'] = proj_xyz
        data_dict['proj_range'] = proj_range
        data_dict['proj_remission'] = proj_remission
        data_dict['proj_range_img'] = proj_range_img
        data_dict['proj_idx'] = proj_idx
        data_dict['proj_x'] = proj_x
        data_dict['proj_y'] = proj_y

        return data_dict


@register_dataset
class point_image_dataset_mix_semkitti(data.Dataset):
    def __init__(self, in_dataset, config, loader_config, num_vote=1, trans_std=[0.1, 0.1, 0.1], max_dropout_ratio=0.2):
        'Initialization'
        self.point_cloud_dataset = in_dataset
        self.config = config
        self.ignore_label = config['dataset_params']['ignore_label']
        self.rotate_aug = loader_config['rotate_aug']
        self.flip_aug = loader_config['flip_aug']
        self.transform = loader_config['transform_aug']
        self.scale_aug = loader_config['scale_aug']
        self.dropout = loader_config['dropout_aug']
        self.instance_aug = loader_config.get('instance_aug', False)
        self.max_volume_space = config['dataset_params']['max_volume_space']
        self.min_volume_space = config['dataset_params']['min_volume_space']
        self.num_vote = num_vote
        self.trans_std = trans_std
        self.max_dropout_ratio = max_dropout_ratio
        self.debug = config['debug']

        self.bottom_crop = config['dataset_params']['bottom_crop']
        color_jitter = config['dataset_params']['color_jitter']
        self.color_jitter = T.ColorJitter(*color_jitter) if color_jitter else None
        self.flip2d = config['dataset_params']['flip2d']
        self.image_normalizer = config['dataset_params']['image_normalizer']
        self.laserscaner = LaserScan()

    def __len__(self):
        'Denotes the total number of samples'
        if self.debug:
            return 100 * self.num_vote
        else:
            return len(self.point_cloud_dataset)

    @staticmethod
    def select_points_in_frustum(points_2d, x1, y1, x2, y2):
        """
        Select points in a 2D frustum parametrized by x1, y1, x2, y2 in image coordinates
        :param points_2d: point cloud projected into 2D
        :param points_3d: point cloud
        :param x1: left bound
        :param y1: upper bound
        :param x2: right bound
        :param y2: lower bound
        :return: points (2D and 3D) that are in the frustum
        """
        keep_ind = (points_2d[:, 0] > x1) * \
                   (points_2d[:, 1] > y1) * \
                   (points_2d[:, 0] < x2) * \
                   (points_2d[:, 1] < y2)

        return keep_ind

    def get_augment_scene(self, index, cut_scene=False):
        'Generates one sample of data'
        data, root = self.point_cloud_dataset[index]

        xyz = data['xyz']
        labels = data['labels']
        instance_label = data['instance_label'].reshape(-1)
        sig = data['signal']
        origin_len = data['origin_len']

        ref_pc = xyz.copy()
        ref_labels = labels.copy()
        ref_index = np.arange(len(ref_pc))

        mask_x = np.logical_and(xyz[:, 0] > self.min_volume_space[0], xyz[:, 0] < self.max_volume_space[0])
        mask_y = np.logical_and(xyz[:, 1] > self.min_volume_space[1], xyz[:, 1] < self.max_volume_space[1])
        mask_z = np.logical_and(xyz[:, 2] > self.min_volume_space[2], xyz[:, 2] < self.max_volume_space[2])
        mask = np.logical_and(mask_x, np.logical_and(mask_y, mask_z))

        if cut_scene:
            mask *= instance_label != 0

        xyz = xyz[mask]
        ref_pc = ref_pc[mask]
        labels = labels[mask]
        instance_label = instance_label[mask]
        ref_index = ref_index[mask]
        sig = sig[mask]
        point_num = len(xyz)

        if self.dropout and self.point_cloud_dataset.imageset == 'train':
            dropout_ratio = np.random.random() * self.max_dropout_ratio
            drop_idx = np.where(np.random.random((xyz.shape[0])) <= dropout_ratio)[0]

            if len(drop_idx) > 0:
                xyz[drop_idx, :] = xyz[0, :]
                labels[drop_idx, :] = labels[0, :]
                sig[drop_idx, :] = sig[0, :]
                instance_label[drop_idx] = instance_label[0]
                ref_index[drop_idx] = ref_index[0]

        # load 2D data
        image = data['img']
        # depth_image = data['depth_img']
        proj_matrix = data['proj_matrix']

        # project points into image
        keep_idx = xyz[:, 0] > 0  # only keep point in front of the vehicle
        points_hcoords = np.concatenate([xyz[keep_idx], np.ones([keep_idx.sum(), 1], dtype=np.float32)], axis=1)
        img_points = (proj_matrix @ points_hcoords.T).T
        img_points = img_points[:, :2] / np.expand_dims(img_points[:, 2], axis=1)  # scale 2D points
        keep_idx_img_pts = self.select_points_in_frustum(img_points, 0, 0, *image.size)
        keep_idx[keep_idx] = keep_idx_img_pts

        # fliplr so that indexing is row, col and not col, row
        img_points = np.fliplr(img_points)
        points_img = img_points[keep_idx_img_pts]

        ### 3D Augmentation ###
        # random data augmentation by rotation
        if self.rotate_aug:
            rotate_rad = np.deg2rad(np.random.random() * 360)
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            xyz[:, :2] = np.dot(xyz[:, :2], j)

        # random data augmentation by flip x , y or x+y
        if self.flip_aug:
            flip_type = np.random.choice(4, 1)
            if flip_type == 1:
                xyz[:, 0] = -xyz[:, 0]
            elif flip_type == 2:
                xyz[:, 1] = -xyz[:, 1]
            elif flip_type == 3:
                xyz[:, :2] = -xyz[:, :2]

        if self.scale_aug:
            noise_scale = np.random.uniform(0.95, 1.05)
            xyz[:, 0] = noise_scale * xyz[:, 0]
            xyz[:, 1] = noise_scale * xyz[:, 1]

        if self.transform:
            noise_translate = np.array([np.random.normal(0, self.trans_std[0], 1),
                                        np.random.normal(0, self.trans_std[1], 1),
                                        np.random.normal(0, self.trans_std[2], 1)]).T

            xyz[:, 0:3] += noise_translate

        img_label = labels[keep_idx]
        point2img_index = np.arange(len(labels))[keep_idx]
        feat = np.concatenate((xyz, sig), axis=1)

        # processing to range image
        self.laserscaner.open_scan_with_points(xyz, sig)
        proj_range = self.laserscaner.proj_range # [H, W]
        proj_xyz = self.laserscaner.proj_xyz # [H, W, 3]
        proj_remission = self.laserscaner.proj_remission # [H, W]
        proj_idx = self.laserscaner.proj_idx # [H, W]
        proj_mask = self.laserscaner.proj_mask # [H, W]
        proj_x = self.laserscaner.proj_x
        proj_y = self.laserscaner.proj_y

        proj_idx = proj_idx[:, :, np.newaxis]
        proj_range = proj_range[:, :, np.newaxis]
        proj_range = np.concatenate([proj_range, proj_range, proj_range], 2)
        proj_range = cv2.normalize(proj_range, None, 0, 1, cv2.NORM_MINMAX)
        proj_xyz = cv2.normalize(proj_xyz, None, 0, 1, cv2.NORM_MINMAX)

        ### 2D Augmentation ###
        if self.bottom_crop:
            # self.bottom_crop is a tuple (crop_width, crop_height)
            left = int(np.random.rand() * (image.size[0] + 1 - self.bottom_crop[0]))
            right = left + self.bottom_crop[0]
            top = image.size[1] - self.bottom_crop[1]
            bottom = image.size[1]

            # update image points
            keep_idx = points_img[:, 0] >= top
            keep_idx = np.logical_and(keep_idx, points_img[:, 0] < bottom)
            keep_idx = np.logical_and(keep_idx, points_img[:, 1] >= left)
            keep_idx = np.logical_and(keep_idx, points_img[:, 1] < right)

            # crop image
            image = image.crop((left, top, right, bottom))
            # depth_image = depth_image.crop((left, top, right, bottom))
            points_img = points_img[keep_idx]
            points_img[:, 0] -= top
            points_img[:, 1] -= left

            img_label = img_label[keep_idx]
            point2img_index = point2img_index[keep_idx]

        img_indices = points_img.astype(np.int64)

        # 2D augmentation
        if self.color_jitter is not None:
            image = self.color_jitter(image)
            # depth_image = self.color_jitter(depth_image)

        # PIL to numpy
        image = np.array(image, dtype=np.float32, copy=False) / 255.
        # depth_image = np.array(depth_image, dtype=np.float32, copy=False) / 255.

        # 2D augmentation
        if np.random.rand() < self.flip2d:
            image = np.ascontiguousarray(np.fliplr(image))
            # depth_image = np.ascontiguousarray(np.fliplr(depth_image))
            img_indices[:, 1] = image.shape[1] - 1 - img_indices[:, 1]

        # normalize image
        if self.image_normalizer:
            mean, std = self.image_normalizer
            mean = np.asarray(mean, dtype=np.float32)
            std = np.asarray(std, dtype=np.float32)
            image = (image - mean) / std

        data_dict = {}
        data_dict['point_feat'] = feat
        data_dict['point_label'] = labels
        data_dict['ref_xyz'] = ref_pc
        data_dict['ref_label'] = ref_labels
        data_dict['ref_index'] = ref_index
        data_dict['mask'] = mask
        data_dict['point_num'] = point_num
        data_dict['origin_len'] = origin_len
        data_dict['root'] = root

        data_dict['img'] = image
        # data_dict['depth_img'] = depth_image
        data_dict['img_indices'] = img_indices
        data_dict['img_label'] = img_label
        data_dict['point2img_index'] = point2img_index

        data_dict['proj_xyz'] = proj_xyz
        data_dict['proj_range'] = proj_range
        data_dict['proj_remission'] = proj_remission
        data_dict['proj_x'] = proj_x
        data_dict['proj_y'] = proj_y

        return data_dict

    def __getitem__(self, index):
        data_dict = self.get_augment_scene(index)

        if self.point_cloud_dataset.imageset == 'train':
            cut_index = random.randint(0, self.__len__() - 1)

            while cut_index == index:
                cut_index = random.randint(0, self.__len__() - 1)

            cut_dict = self.get_augment_scene(cut_index, cut_scene=True)
            cutmix_data_dict = {}
            for keys in data_dict.keys():
                if keys == 'point_num' or keys == 'origin_len':
                    cutmix_data_dict[keys] = data_dict[keys] + cut_dict[keys]
                elif keys == 'ref_index':
                    cut_dict[keys] = cut_dict[keys] + data_dict['origin_len']
                    cutmix_data_dict[keys] = np.append(data_dict[keys], cut_dict[keys])
                elif keys == 'mask':
                    cutmix_data_dict[keys] = np.append(data_dict[keys], cut_dict[keys])
                elif keys not in ['img', 'img_indices', 'img_label', 'point2img_index', 'depth_img', 'proj_xyz', 'proj_range', 'proj_x', 'proj_y']:                   
                    cutmix_data_dict[keys] = np.vstack((data_dict[keys], cut_dict[keys]))
                else:
                    cutmix_data_dict[keys] = data_dict[keys]
            raw_points, cut_points = data_dict['point_feat'], cut_dict['point_feat']
            points = np.concatenate((raw_points, cut_points), 0)
            # processing to range image
            self.laserscaner.open_scan_with_points(points[:, 0:3], points[:, -1].reshape(-1, 1))
            proj_range = self.laserscaner.proj_range # [H, W]
            proj_xyz = self.laserscaner.proj_xyz # [H, W, 3]
            proj_idx = self.laserscaner.proj_idx # [H, W]
            proj_x = self.laserscaner.proj_x
            proj_y = self.laserscaner.proj_y

            proj_idx = proj_idx[:, :, np.newaxis]
            proj_range = proj_range[:, :, np.newaxis]
            proj_remission = proj_remission[:, :, np.newaxis]
            proj_range_img = np.concatenate([proj_xyz, proj_range, proj_remission], 2)
            proj_range = np.concatenate([proj_range, proj_range, proj_range], 2)
            proj_remission = np.concatenate([proj_remission, proj_remission, proj_remission], 2)
            proj_range = cv2.normalize(proj_range, None, 0, 1, cv2.NORM_MINMAX)
            proj_remission = cv2.normalize(proj_remission, None, 0, 1, cv2.NORM_MINMAX)
            proj_xyz = cv2.normalize(proj_xyz, None, 0, 1, cv2.NORM_MINMAX)
            proj_range_img = cv2.normalize(proj_range_img, None, 0, 1, cv2.NORM_MINMAX)
            cutmix_data_dict['proj_xyz'] = proj_xyz
            cutmix_data_dict['proj_range'] = proj_range
            cutmix_data_dict['proj_range_img'] = proj_range_img
            cutmix_data_dict['proj_remission'] = proj_remission
            cutmix_data_dict['proj_x'] = proj_x
            cutmix_data_dict['proj_y'] = proj_y

        else:
            cutmix_data_dict = data_dict

        return cutmix_data_dict
@register_dataset
class point_image_dataset_range_mae_kitti(data.Dataset):
    def __init__(self, in_dataset, config, loader_config, num_vote=1, trans_std=[0.1, 0.1, 0.1], max_dropout_ratio=0.2):
        'Initialization'
        self.point_cloud_dataset = in_dataset
        self.config = config
        self.ignore_label = config['dataset_params']['ignore_label']
        self.rotate_aug = loader_config['rotate_aug']
        self.flip_aug = loader_config['flip_aug']
        self.transform = loader_config['transform_aug']
        self.scale_aug = loader_config['scale_aug']
        self.dropout = loader_config['dropout_aug']
        self.instance_aug = loader_config.get('instance_aug', False)
        self.max_volume_space = config['dataset_params']['max_volume_space']
        self.min_volume_space = config['dataset_params']['min_volume_space']
        self.num_vote = num_vote
        self.trans_std = trans_std
        self.max_dropout_ratio = max_dropout_ratio
        self.debug = config['debug']

        self.bottom_crop = config['dataset_params']['bottom_crop']
        color_jitter = config['dataset_params']['color_jitter']
        self.color_jitter = T.ColorJitter(*color_jitter) if color_jitter else None
        self.flip2d = config['dataset_params']['flip2d']
        self.image_normalizer = config['dataset_params']['image_normalizer']
        self.laserscaner = LaserScan(H=64, W=2048, fov_up=3.0, fov_down=-25.0)


    def __len__(self):
        'Denotes the total number of samples'
        if self.debug:
            return 100 * self.num_vote
        else:
            return len(self.point_cloud_dataset)

    @staticmethod
    def select_points_in_frustum(points_2d, x1, y1, x2, y2):
        """
        Select points in a 2D frustum parametrized by x1, y1, x2, y2 in image coordinates
        :param points_2d: point cloud projected into 2D
        :param points_3d: point cloud
        :param x1: left bound
        :param y1: upper bound
        :param x2: right bound
        :param y2: lower bound
        :return: points (2D and 3D) that are in the frustum
        """
        keep_ind = (points_2d[:, 0] > x1) * \
                   (points_2d[:, 1] > y1) * \
                   (points_2d[:, 0] < x2) * \
                   (points_2d[:, 1] < y2)

        return keep_ind

    def __getitem__(self, index):
        data_dict = {}
        'Generates one sample of data'
        data, root = self.point_cloud_dataset[index]

        xyz = data['xyz']
        labels = data['labels']
        instance_label = data['instance_label'].reshape(-1)
        sig = data['signal']
        origin_len = data['origin_len']

        ref_pc = xyz.copy()
        ref_labels = labels.copy()
        ref_index = np.arange(len(ref_pc))

        mask_x = np.logical_and(xyz[:, 0] > self.min_volume_space[0], xyz[:, 0] < self.max_volume_space[0])
        mask_y = np.logical_and(xyz[:, 1] > self.min_volume_space[1], xyz[:, 1] < self.max_volume_space[1])
        mask_z = np.logical_and(xyz[:, 2] > self.min_volume_space[2], xyz[:, 2] < self.max_volume_space[2])
        mask = np.logical_and(mask_x, np.logical_and(mask_y, mask_z))

        xyz = xyz[mask]
        ref_pc = ref_pc[mask]
        labels = labels[mask]
        instance_label = instance_label[mask]
        ref_index = ref_index[mask]
        sig = sig[mask]
        point_num = len(xyz)

        if self.dropout and self.point_cloud_dataset.imageset == 'train':
            dropout_ratio = np.random.random() * self.max_dropout_ratio
            drop_idx = np.where(np.random.random((xyz.shape[0])) <= dropout_ratio)[0]

            if len(drop_idx) > 0:
                xyz[drop_idx, :] = xyz[0, :]
                labels[drop_idx, :] = labels[0, :]
                sig[drop_idx, :] = sig[0, :]
                instance_label[drop_idx] = instance_label[0]
                ref_index[drop_idx] = ref_index[0]

        # load 2D data
        image = data['img']
        # depth_image = data['depth_img']
        proj_matrix = data['proj_matrix']

        # project points into image
        keep_idx = xyz[:, 0] > 0  # only keep point in front of the vehicle
        points_hcoords = np.concatenate([xyz[keep_idx], np.ones([keep_idx.sum(), 1], dtype=np.float32)], axis=1)
        img_points = (proj_matrix @ points_hcoords.T).T
        img_points = img_points[:, :2] / np.expand_dims(img_points[:, 2], axis=1)  # scale 2D points
        keep_idx_img_pts = self.select_points_in_frustum(img_points, 0, 0, *image.size)
        keep_idx[keep_idx] = keep_idx_img_pts

        # fliplr so that indexing is row, col and not col, row
        img_points = np.fliplr(img_points)
        points_img = img_points[keep_idx_img_pts]


        ### 3D Augmentation ###
        # random data augmentation by rotation
        if self.rotate_aug:
            rotate_rad = np.deg2rad(np.random.random() * 360)
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            xyz[:, :2] = np.dot(xyz[:, :2], j)

        # random data augmentation by flip x , y or x+y
        if self.flip_aug:
            flip_type = np.random.choice(4, 1)
            if flip_type == 1:
                xyz[:, 0] = -xyz[:, 0]
            elif flip_type == 2:
                xyz[:, 1] = -xyz[:, 1]
            elif flip_type == 3:
                xyz[:, :2] = -xyz[:, :2]

        if self.scale_aug:
            noise_scale = np.random.uniform(0.95, 1.05)
            xyz[:, 0] = noise_scale * xyz[:, 0]
            xyz[:, 1] = noise_scale * xyz[:, 1]

        if self.transform:
            noise_translate = np.array([np.random.normal(0, self.trans_std[0], 1),
                                        np.random.normal(0, self.trans_std[1], 1),
                                        np.random.normal(0, self.trans_std[2], 1)]).T

            xyz[:, 0:3] += noise_translate

        img_label = labels[keep_idx]
        point2img_index = np.arange(len(labels))[keep_idx]
        feat = np.concatenate((xyz, sig), axis=1)

        # processing to range image
        out_dict = self.laserscaner.open_scan(feat)
        data_dict['laser_range'] = out_dict['range']
        data_dict['laser_ori_xyz'] = out_dict['ori_xyz']
        data_dict['laser_ori_r'] = out_dict['ori_r']
        data_dict['laser_idx'] = out_dict['idx']
        data_dict['laser_mask'] = out_dict['mask']
        data_dict['laser_range_in'] = out_dict['range_in']
        data_dict['laser_y'] = out_dict['y']
        data_dict['laser_x'] = out_dict['x']
        data_dict['laser_points'] = out_dict['points']

        points_xyz = out_dict['points']
        # print('points_xyz:,', points_xyz.shape)
        tree = kdtree(points_xyz)
        _, knns = tree.query(points_xyz, k=7)

        data_dict['knns'] = knns
        data_dict['num_points'] = points_xyz.shape[0]
        ### 2D Augmentation ###
        if self.bottom_crop:
            # crop image for processing:
            # left = ( image.size[0] - self.bottom_crop[0] ) // 2
            left = int(np.random.rand() * (image.size[0] + 1 - self.bottom_crop[0]))
            right = left + self.bottom_crop[0] 
            top = image.size[1] - self.bottom_crop[1]
            bottom = image.size[1]
            
            # update image points
            keep_idx = points_img[:, 0] >= top
            keep_idx = np.logical_and(keep_idx, points_img[:, 0] < bottom)
            keep_idx = np.logical_and(keep_idx, points_img[:, 1] >= left)
            keep_idx = np.logical_and(keep_idx, points_img[:, 1] < right)

            # crop image
            # image = image[top:bottom, left:right, :]
            image = image.crop((left, top, right, bottom))
            points_img = points_img[keep_idx]
            points_img[:, 0] -= top
            points_img[:, 1] -= left

            img_label = img_label[keep_idx]
            point2img_index = point2img_index[keep_idx]

        img_indices = points_img.astype(np.int64)

        # 2D augmentation
        if self.color_jitter is not None:
            image = self.color_jitter(image)
            # depth_image = self.color_jitter(depth_image)

        # PIL to numpy
        image = np.array(image, dtype=np.float32, copy=False) / 255.
        # depth_image = np.array(depth_image, dtype=np.float32, copy=False) / 255.

        # 2D augmentation
        if np.random.rand() < self.flip2d:
            image = np.ascontiguousarray(np.fliplr(image))
            # depth_image = np.ascontiguousarray(np.fliplr(depth_image))
            img_indices[:, 1] = image.shape[1] - 1 - img_indices[:, 1]

        # normalize image
        if self.image_normalizer:
            mean, std = self.image_normalizer
            mean = np.asarray(mean, dtype=np.float32)
            std = np.asarray(std, dtype=np.float32)
            image = (image - mean) / std
            # depth_image = (depth_image - mean) / std

        data_dict['point_feat'] = feat
        data_dict['point_label'] = labels
        data_dict['ref_xyz'] = ref_pc
        data_dict['ref_label'] = ref_labels
        data_dict['ref_index'] = ref_index
        data_dict['mask'] = mask
        data_dict['point_num'] = point_num
        data_dict['origin_len'] = origin_len
        data_dict['root'] = root

        data_dict['img'] = image
        # data_dict['depth_img'] = depth_image
        data_dict['img_indices'] = img_indices
        data_dict['img_label'] = img_label
        data_dict['point2img_index'] = point2img_index

        t_or_f_range_img = make_random_range_mask(mask_ratio=0)
        
        t_or_f_point = t_or_f_range_img[out_dict['y'].astype(np.int16), out_dict['x'].astype(np.int16)]
        data_dict['sample_points'] = feat[t_or_f_point]
        data_dict['sample_index'] = np.arange(len(feat))[t_or_f_point]
        data_dict['unsample_index'] = np.arange(len(feat))[np.invert(t_or_f_point)]
        data_dict['spconv_points'] = feat

        return data_dict




@register_dataset
class point_image_dataset_nus(data.Dataset):
    def __init__(self, in_dataset, config, loader_config, num_vote=1, trans_std=[0.1, 0.1, 0.1], max_dropout_ratio=0.2):
        'Initialization'
        self.point_cloud_dataset = in_dataset
        self.config = config
        self.ignore_label = config['dataset_params']['ignore_label']
        self.rotate_aug = loader_config['rotate_aug']
        self.flip_aug = loader_config['flip_aug']
        self.transform = loader_config['transform_aug']
        self.scale_aug = loader_config['scale_aug']
        self.dropout = loader_config['dropout_aug']
        self.instance_aug = loader_config.get('instance_aug', False)
        self.max_volume_space = config['dataset_params']['max_volume_space']
        self.min_volume_space = config['dataset_params']['min_volume_space']
        self.num_vote = num_vote
        self.trans_std = trans_std
        self.max_dropout_ratio = max_dropout_ratio
        self.debug = config['debug']

        self.resize = config['dataset_params'].get('resize', False)
        color_jitter = config['dataset_params']['color_jitter']
        self.color_jitter = T.ColorJitter(*color_jitter) if color_jitter else None
        self.flip2d = config['dataset_params']['flip2d']
        self.image_normalizer = config['dataset_params'].get('image_normalizer', False)

    def map_pointcloud_to_image(self, pc, im_shape, info):
        """
        Maps the lidar point cloud to the image.
        :param pc: (3, N)
        :param im_shape: image to check size and debug
        :param info: dict with calibration infos
        :param im: image, only for visualization
        :return:
        """
        pc = pc.copy().T

        # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
        # First step: transform the point-cloud to the ego vehicle frame for the timestamp of the sweep.
        pc = Quaternion(info['lidar2ego_rotation']).rotation_matrix @ pc
        pc = pc + np.array(info['lidar2ego_translation'])[:, np.newaxis]

        # Second step: transform to the global frame.
        pc = Quaternion(info['ego2global_rotation_lidar']).rotation_matrix @ pc
        pc = pc + np.array(info['ego2global_translation_lidar'])[:, np.newaxis]

        # Third step: transform into the ego vehicle frame for the timestamp of the image.
        pc = pc - np.array(info['ego2global_translation_cam'])[:, np.newaxis]
        pc = Quaternion(info['ego2global_rotation_cam']).rotation_matrix.T @ pc

        # Fourth step: transform into the camera.
        pc = pc - np.array(info['cam2ego_translation'])[:, np.newaxis]
        pc = Quaternion(info['cam2ego_rotation']).rotation_matrix.T @ pc

        # Fifth step: actually take a "picture" of the point cloud.
        # Grab the depths (camera frame z axis points away from the camera).
        depths = pc[2, :]

        # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
        points = view_points(pc, np.array(info['cam_intrinsic']), normalize=True)

        # Cast to float32 to prevent later rounding errors
        points = points.astype(np.float32)

        # Remove points that are either outside or behind the camera.
        mask = np.ones(depths.shape[0], dtype=bool)
        mask = np.logical_and(mask, depths > 0)
        mask = np.logical_and(mask, points[0, :] > 0)
        mask = np.logical_and(mask, points[0, :] < im_shape[1])
        mask = np.logical_and(mask, points[1, :] > 0)
        mask = np.logical_and(mask, points[1, :] < im_shape[0])

        return mask, pc.T, points.T[:, :2]

    def __len__(self):
        'Denotes the total number of samples'
        if self.debug:
            return 100 * self.num_vote
        else:
            return len(self.point_cloud_dataset)

    def __getitem__(self, index):
        'Generates one sample of data'
        data, root = self.point_cloud_dataset[index]

        xyz = data['xyz']
        labels = data['labels']
        sig = data['signal']
        origin_len = data['origin_len']

        # load 2D data
        image = data['img']
        calib_infos = data['calib_infos']

        ref_pc = xyz.copy()
        ref_labels = labels.copy()
        ref_index = np.arange(len(ref_pc))

        mask_x = np.logical_and(xyz[:, 0] > self.min_volume_space[0], xyz[:, 0] < self.max_volume_space[0])
        mask_y = np.logical_and(xyz[:, 1] > self.min_volume_space[1], xyz[:, 1] < self.max_volume_space[1])
        mask_z = np.logical_and(xyz[:, 2] > self.min_volume_space[2], xyz[:, 2] < self.max_volume_space[2])
        mask = np.logical_and(mask_x, np.logical_and(mask_y, mask_z))

        xyz = xyz[mask]
        ref_pc = ref_pc[mask]
        labels = labels[mask]
        ref_index = ref_index[mask]
        sig = sig[mask]
        point_num = len(xyz)

        # dropout points
        if self.dropout and self.point_cloud_dataset.imageset == 'train':
            dropout_ratio = np.random.random() * self.max_dropout_ratio
            drop_idx = np.where(np.random.random((xyz.shape[0])) <= dropout_ratio)[0]

            if len(drop_idx) > 0:
                xyz[drop_idx, :] = xyz[0, :]
                labels[drop_idx, :] = labels[0, :]
                sig[drop_idx, :] = sig[0, :]
                ref_index[drop_idx] = ref_index[0]

        keep_idx, _, points_img = self.map_pointcloud_to_image(
            xyz, (image.size[1], image.size[0]), calib_infos)
        points_img = np.ascontiguousarray(np.fliplr(points_img))

        # random data augmentation by rotation
        if self.rotate_aug:
            rotate_rad = np.deg2rad(np.random.random() * 360)
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            xyz[:, :2] = np.dot(xyz[:, :2], j)

        # random data augmentation by flip x , y or x+y
        if self.flip_aug:
            flip_type = np.random.choice(4, 1)
            if flip_type == 1:
                xyz[:, 0] = -xyz[:, 0]
            elif flip_type == 2:
                xyz[:, 1] = -xyz[:, 1]
            elif flip_type == 3:
                xyz[:, :2] = -xyz[:, :2]

        if self.scale_aug:
            noise_scale = np.random.uniform(0.95, 1.05)
            xyz[:, 0] = noise_scale * xyz[:, 0]
            xyz[:, 1] = noise_scale * xyz[:, 1]

        if self.transform:
            noise_translate = np.array([np.random.normal(0, self.trans_std[0], 1),
                                        np.random.normal(0, self.trans_std[1], 1),
                                        np.random.normal(0, self.trans_std[2], 1)]).T

            xyz[:, 0:3] += noise_translate

        points_img = points_img[keep_idx]
        img_label = labels[keep_idx]
        point2img_index = np.arange(len(keep_idx))[keep_idx]
        feat = np.concatenate((xyz, sig), axis=1)

        ### 2D Augmentation ###
        if self.resize:
            assert image.size[0] > self.resize[0]

            # scale image points
            points_img[:, 0] = float(self.resize[1]) / image.size[1] * np.floor(points_img[:, 0])
            points_img[:, 1] = float(self.resize[0]) / image.size[0] * np.floor(points_img[:, 1])

            # resize image
            image = image.resize(self.resize, Image.BILINEAR)

        img_indices = points_img.astype(np.int64)

        # 2D augmentation
        if self.color_jitter is not None:
            image = self.color_jitter(image)

        image = np.array(image, dtype=np.float32, copy=False) / 255.

        # 2D augmentation
        if np.random.rand() < self.flip2d:
            image = np.ascontiguousarray(np.fliplr(image))
            img_indices[:, 1] = image.shape[1] - 1 - img_indices[:, 1]

        # normalize image
        if self.image_normalizer:
            mean, std = self.image_normalizer
            mean = np.asarray(mean, dtype=np.float32)
            std = np.asarray(std, dtype=np.float32)
            image = (image - mean) / std

        data_dict = {}
        data_dict['point_feat'] = feat
        data_dict['point_label'] = labels
        data_dict['ref_xyz'] = ref_pc
        data_dict['ref_label'] = ref_labels
        data_dict['ref_index'] = ref_index
        data_dict['mask'] = mask
        data_dict['point_num'] = point_num
        data_dict['origin_len'] = origin_len
        data_dict['root'] = root

        data_dict['img'] = image
        data_dict['img_indices'] = img_indices
        data_dict['img_label'] = img_label
        data_dict['point2img_index'] = point2img_index

        return data_dict


@register_dataset
class voxel_dataset(data.Dataset):
    def __init__(self, in_dataset, config, loader_config, num_vote=1, trans_std=[0.1, 0.1, 0.1], max_dropout_ratio=0.2):
        'Initialization'
        self.point_cloud_dataset = in_dataset
        self.config = config
        self.ignore_label = config['dataset_params']['ignore_label']
        self.rotate_aug = loader_config['rotate_aug']
        self.flip_aug = loader_config['flip_aug']
        self.transform = loader_config['transform_aug']
        self.scale_aug = loader_config['scale_aug']
        self.dropout = loader_config['dropout_aug']
        self.voxel_size = config['model_params']['voxel_size']
        self.num_vote = num_vote
        self.trans_std = trans_std
        self.max_dropout_ratio = max_dropout_ratio
        self.debug = config['debug']

    def __len__(self):
        'Denotes the total number of samples'
        if self.debug:
            return 100 * self.num_vote
        else:
            return len(self.point_cloud_dataset)


    def __getitem__(self, index):
        'Generates one sample of data'
        data, root = self.point_cloud_dataset[index]

        xyz = data['xyz']
        labels = data['labels']
        sig = data['signal']
        origin_len = data['origin_len']

        # random data augmentation by rotation
        if self.rotate_aug:
            rotate_rad = np.deg2rad(np.random.random() * 360)
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            xyz[:, :2] = np.dot(xyz[:, :2], j)

        # random data augmentation by flip x , y or x+y
        if self.flip_aug:
            flip_type = np.random.choice(4, 1)
            if flip_type == 1:
                xyz[:, 0] = -xyz[:, 0]
            elif flip_type == 2:
                xyz[:, 1] = -xyz[:, 1]
            elif flip_type == 3:
                xyz[:, :2] = -xyz[:, :2]

        if self.scale_aug:
            noise_scale = np.random.uniform(0.95, 1.05)
            xyz[:, 0] = noise_scale * xyz[:, 0]
            xyz[:, 1] = noise_scale * xyz[:, 1]

        if self.transform:
            noise_translate = np.array([np.random.normal(0, self.trans_std[0], 1),
                                        np.random.normal(0, self.trans_std[1], 1),
                                        np.random.normal(0, self.trans_std[2], 1)]).T

            xyz[:, 0:3] += noise_translate

        if self.dropout and self.point_cloud_dataset.imageset == 'train':
            dropout_ratio = np.random.random() * self.max_dropout_ratio
            drop_idx = np.where(np.random.random((xyz.shape[0])) <= dropout_ratio)[0]

            if len(drop_idx) > 0:
                xyz[drop_idx, :] = xyz[0, :]
                labels[drop_idx, :] = labels[0, :]
                sig[drop_idx, :] = sig[0, :]

        ref_pc = xyz.copy()
        ref_labels = labels.copy()
        ref_index = np.arange(len(ref_pc))
        pc_ = np.round(xyz / self.voxel_size)
        pc_ = pc_ - pc_.min(0, keepdims=1)
        feat_ = np.concatenate((xyz, sig), axis=1)

        _, inds, inverse_map = sparse_quantize(pc_, 1, return_index=True, return_inverse=True)

        pc = pc_[inds]
        feat = feat_[inds]
        labels = labels[inds]
        num_voxel = len(inds)
        points = SparseTensor(ref_pc, pc_)
        ref_index = SparseTensor(ref_index, pc_)
        map = SparseTensor(inds, pc)
        lidar = SparseTensor(feat, pc)
        labels = SparseTensor(labels, pc)
        labels_mapped = SparseTensor(ref_labels, pc_)
        inverse_map = SparseTensor(inverse_map, pc_)

        data_dict = {}
        data_dict['lidar'] = lidar
        data_dict['points'] = points
        data_dict['targets'] = labels
        data_dict['targets_mapped'] = labels_mapped
        data_dict['ref_index'] = ref_index
        data_dict['origin_len'] = origin_len
        data_dict['root'] = root
        data_dict['map'] = map
        data_dict['num_voxel'] = num_voxel
        data_dict['inverse_map'] = inverse_map

        return data_dict


@register_collate_fn
def collate_fn_default(data):
    point_num = [d['point_num'] for d in data]
    batch_size = len(point_num)
    ref_labels = data[0]['ref_label']
    origin_len = data[0]['origin_len']
    ref_indices = [torch.from_numpy(d['ref_index']) for d in data]
    point2img_index = [torch.from_numpy(d['point2img_index']).long() for d in data]
    path = [d['root'] for d in data]

    img = [torch.from_numpy(d['img']) for d in data]
    laser_range_in = [torch.from_numpy(d['laser_range_in']) for d in data]
    # proj_xyz = [torch.from_numpy(d['proj_xyz']) for d in data]
    # proj_range = [torch.from_numpy(d['proj_range']) for d in data]
    # proj_range_img = [torch.from_numpy(d['proj_range_img']) for d in data]
    # proj_idx = [torch.from_numpy(d['proj_idx']) for d in data]
    img_indices = [d['img_indices'] for d in data]
    img_label = [torch.from_numpy(d['img_label']) for d in data]

    b_idx = []
    coord_x_pad, coord_y_pad = [], []
    for i in range(batch_size):
        proj_x = data[i]['laser_x'].reshape(-1, 1)
        proj_y = data[i]['laser_y'].reshape(-1, 1)
        coor_x = np.concatenate([i*np.ones((proj_x.shape[0], 1)), proj_x], -1)
        coor_y = np.concatenate([i*np.ones((proj_y.shape[0], 1)), proj_y], -1)
        coord_x_pad.append(torch.from_numpy(coor_x))
        coord_y_pad.append(torch.from_numpy(coor_y))
        b_idx.append(torch.ones(point_num[i]) * i)
    
    sample_idx = [torch.from_numpy(d['sample_index'].astype(np.int)) for d in data]
    coors_sample, coors_spconv = [], []
    batch_idx_sample = []
    for i in range(batch_size):
        coor_sample_points = data[i]['sample_points']
        coor_spconv_points = data[i]['spconv_points']
        batch_idx_sample.append(torch.ones(len(coor_spconv_points)) * i)
        coor_pad_sample = np.pad(coor_sample_points, ((0,0), (1,0)), mode='constant', constant_values=i)
        coor_pad_spconv = np.pad(coor_spconv_points, ((0,0), (1,0)), mode='constant', constant_values=i)
        coors_sample.append(torch.from_numpy(coor_pad_sample))
        coors_spconv.append(torch.from_numpy(coor_pad_spconv))
    
    points = [torch.from_numpy(d['point_feat']) for d in data]
    spconv_points = [torch.from_numpy(d['spconv_points']) for d in data]
    sample_points = [torch.from_numpy(d['sample_points']) for d in data]
    ref_xyz = [torch.from_numpy(d['ref_xyz']) for d in data]
    labels = [torch.from_numpy(d['point_label']) for d in data]
    knns = [torch.from_numpy(d['knns']) for d in data]

    num_points = [d['num_points'] for d in data]

    return {
        'sample_points_batch_idx': torch.cat(coors_sample, 0),
        'spconv_points_batch_idx': torch.cat(coors_spconv, 0),

        'points': torch.cat(points).float(),
        'spconv_points': torch.cat(spconv_points).float(),
        'sample_points': torch.cat(sample_points).float(),
        'ref_xyz': torch.cat(ref_xyz).float(),
        'batch_idx': torch.cat(b_idx).long(),
        'batch_idx_sample': torch.cat(batch_idx_sample).long(),
        'batch_size': batch_size,
        'labels': torch.cat(labels).long().squeeze(1),
        'raw_labels': torch.from_numpy(ref_labels).long(),
        'origin_len': origin_len,
        'indices': torch.cat(ref_indices).long(),
        'point2img_index': point2img_index,
        'img': torch.stack(img, 0).permute(0, 3, 1, 2),
        'knns': torch.cat(knns, 0),
        'num_points': torch.LongTensor(num_points),
        # 'depth_img': torch.stack(depth_img, 0).permute(0, 3, 1, 2),
        # 'proj_xyz': torch.stack(proj_xyz, 0).permute(0, 3, 1, 2),
        # 'proj_range': torch.stack(proj_range, 0).permute(0, 3, 1, 2),
        # 'proj_range_img': torch.stack(proj_range_img, 0).permute(0, 3, 1, 2),
        'laser_range_in': torch.stack(laser_range_in, 0),
        'laser_x': torch.cat(coord_x_pad, 0).long(),
        'laser_y': torch.cat(coord_y_pad, 0).long(),
        'img_indices': img_indices,
        'points_img': img_indices,
        'img_label': torch.cat(img_label, 0).squeeze(1).long(),
        'path': path,
        'sample_index': sample_idx,
    }


@register_collate_fn
def collate_fn_voxel(inputs):
    return sparse_collate_fn(inputs)