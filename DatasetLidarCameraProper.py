# -------------------------------------------------------------------
# Copyright (C) 2020 Università degli studi di Milano-Bicocca, iralab
# Author: Daniele Cattaneo (d.cattaneo10@campus.unimib.it)
# Released under Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# http://creativecommons.org/licenses/by-nc-sa/4.0/
# -------------------------------------------------------------------

# Modified Author: Xudong Lv
# based on github.com/cattaneod/CMRNet/blob/master/DatasetVisibilityKitti.py

import csv
import os
from math import radians
import cv2

import h5py
import mathutils
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TTF
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils import invert_pose, rotate_forward, quaternion_from_matrix
from pykitti import odometry
import pykitti


class DatasetLidarCameraKittiOdometry(Dataset):

    def __init__(self, dataset_dir, transform=None, augmentation=False, use_reflectance=False,
                 max_t=1.5, max_r=20., split='val', device='cpu', val_sequence='00', suf='.png'):
        super(DatasetLidarCameraKittiOdometry, self).__init__()

        self.use_reflectance = use_reflectance
        self.maps_folder = ''
        self.device = device
        self.max_r = max_r
        self.max_t = max_t
        self.augmentation = augmentation
        self.root_dir = dataset_dir
        self.transform = transform
        self.split = split
        self.GTs_R = {}
        self.GTs_T = {}
        self.GTs_T_cam02_velo = {}
        self.K = {}
        self.suf = suf

        self.all_files = []
        self.sequence_list = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
                              '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']

        for seq in self.sequence_list:
            odom = odometry(self.root_dir, seq)

            # Save calibration matrices for each sequence
            self.K[seq] = odom.calib.K_cam2 # 3x3
            self.GTs_T_cam02_velo[seq] = odom.calib.T_cam2_velo # velodyne to rectified camera coordinate transform (T_cam02_velo: 4x4)

            # Build list of paths images and pointclouds
            image_list = os.listdir(os.path.join(dataset_dir, 'sequences', seq, 'image_2'))
            image_list.sort()

            for image_name in image_list:
                # Skip if not both, image and pointcloud are present
                if not os.path.exists(os.path.join(dataset_dir, 'sequences', seq, 'velodyne',
                                                   str(image_name.split('.')[0])+'.bin')):
                    continue
                if not os.path.exists(os.path.join(dataset_dir, 'sequences', seq, 'image_2',
                                                   str(image_name.split('.')[0])+suf)):
                    continue

                # Some bullshit
                if seq == val_sequence:
                    if split.startswith('val') or split == 'test': # if it is a validation split, append it to all files
                        self.all_files.append(os.path.join(seq, image_name.split('.')[0]))
                elif (not seq == val_sequence) and split == 'train': # if it is a train split, still append lol
                    self.all_files.append(os.path.join(seq, image_name.split('.')[0]))

        # Create RT files for validation sequences, whatever that is
        self.val_RT = []
        if split == 'val' or split == 'test':
            val_RT_file = os.path.join(dataset_dir, 'sequences',
                                       f'val_RT_left_seq{val_sequence}_{max_r:.2f}_{max_t:.2f}.csv')
            if os.path.exists(val_RT_file):
                print(f'VAL SET: Using this file: {val_RT_file}')
                df_test_RT = pd.read_csv(val_RT_file, sep=',')
                for index, row in df_test_RT.iterrows():
                    self.val_RT.append(list(row))
            else:
                print(f'VAL SET - Not found: {val_RT_file}')
                print("Generating a new one")
                val_RT_file = open(val_RT_file, 'w')
                val_RT_file = csv.writer(val_RT_file, delimiter=',')
                val_RT_file.writerow(['id', 'tx', 'ty', 'tz', 'rx', 'ry', 'rz'])
                for i in range(len(self.all_files)):
                    rotz = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    roty = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    rotx = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    transl_x = np.random.uniform(-max_t, max_t)
                    transl_y = np.random.uniform(-max_t, max_t)
                    transl_z = np.random.uniform(-max_t, max_t)

                    val_RT_file.writerow([i, transl_x, transl_y, transl_z,
                                           rotx, roty, rotz])
                    self.val_RT.append([float(i), float(transl_x), float(transl_y), float(transl_z),
                                         float(rotx), float(roty), float(rotz)])

            assert len(self.val_RT) == len(self.all_files), "Something wrong with test RTs"
    
    # Retrieves ground truth poses  
    def get_ground_truth_poses(self, sequence, frame):
        return self.GTs_T[sequence][frame], self.GTs_R[sequence][frame]

    # Applies a transformation to the rgb image
    def custom_transform(self, rgb, img_rotation=0., flip=False):
        to_tensor = transforms.ToTensor()
        normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

        if self.split == 'train':
            color_transform = transforms.ColorJitter(0.1, 0.1, 0.1)
            rgb = color_transform(rgb)
            if flip:
                rgb = TTF.hflip(rgb)
            rgb = TTF.rotate(rgb, img_rotation)

        rgb = to_tensor(rgb)
        rgb = normalization(rgb)
        return rgb

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        item = self.all_files[idx] # All files contains paths to all samples in the split (over all sequences in the split)
        seq = str(item.split('/')[0])
        rgb_name = str(item.split('/')[1])

        img_path = os.path.join(self.root_dir, 'sequences', seq, 'image_2', rgb_name+self.suf)
        lidar_path = os.path.join(self.root_dir, 'sequences', seq, 'velodyne', rgb_name+'.bin')
        lidar2_path = os.path.join(self.root_dir, 'sequences', seq, 'velodyne', rgb_name+'.bin') # TODO: change to lidar_ and lidar_r

        lidar_scan = np.fromfile(lidar_path, dtype=np.float32) # load to numpy array from file
        lidar_scan2 = np.fromfile(lidar2_path, dtype=np.float32) # load to numpy array from file

        pc = lidar_scan.reshape((-1, 4))
        pc2 = lidar_scan2.reshape((-1, 4))

        # Clip point-cloud
        valid_indices = pc[:, 0] < -3.
        valid_indices = valid_indices | (pc[:, 0] > 3.)
        valid_indices = valid_indices | (pc[:, 1] < -3.)
        valid_indices = valid_indices | (pc[:, 1] > 3.)
        pc = pc[valid_indices].copy()
        pc_org = torch.from_numpy(pc.astype(np.float32)) # make torch tensor

        valid_indices2 = pc2[:, 0] < -3.
        valid_indices2 = valid_indices2 | (pc2[:, 0] > 3.)
        valid_indices2 = valid_indices2 | (pc2[:, 1] < -3.)
        valid_indices2 = valid_indices2 | (pc2[:, 1] > 3.)
        pc2 = pc2[valid_indices2].copy()
        pc2_org = torch.from_numpy(pc2.astype(np.float32)) # make torch tensor

        RT = self.GTs_T_cam02_velo[seq].astype(np.float32) # TODO: transformation will be different once we have two lidars

        if pc_org.shape[1] == 4 or pc_org.shape[1] == 3: # Make pointcloud 3XN tensor instead of Nx3
            pc_org = pc_org.t()
            pc2_org = pc2_org.t()
        if pc_org.shape[0] == 3: # Force homogenous
            homogeneous = torch.ones(pc_org.shape[1]).unsqueeze(0)
            pc_org = torch.cat((pc_org, homogeneous), 0)
            homogeneous2 = torch.ones(pc2_org.shape[1]).unsqueeze(0)
            pc2_org = torch.cat((pc2_org, homogeneous2), 0)

        elif pc_org.shape[0] == 4: # Force homogenous
            if not torch.all(pc_org[3, :] == 1.):
                pc_org[3, :] = 1.
            if not torch.all(pc2_org[3, :] == 1.):
                pc2_org[3, :] = 1.
        else:
            raise TypeError("Wrong PointCloud shape")
        
        pc_rot = np.matmul(RT, pc_org.numpy()) # move points to camera coordinates, rotation and translation
        pc_rot = pc_rot.astype(np.float32).copy()
        pc_in = torch.from_numpy(pc_rot)

        pc2_rot = np.matmul(RT, pc2_org.numpy()) # move points to camera coordinates, rotation and translation TODO: change to RT2
        pc2_rot = pc2_rot.astype(np.float32).copy()
        pc2_in = torch.from_numpy(pc2_rot)

        h_mirror = False

        img = Image.open(img_path)
        img_rotation = 0.

        try:
            img = self.custom_transform(img, img_rotation, h_mirror)
        except OSError:
            new_idx = np.random.randint(0, self.__len__())
            return self.__getitem__(new_idx)

        # Rotate PointCloud for img_rotation TODO: do for RT and RT2
        if self.split == 'train': # does not do anything since all values are 0???
            R = mathutils.Euler((radians(img_rotation), 0, 0), 'XYZ')
            T = mathutils.Vector((0., 0., 0.))
            pc_in = rotate_forward(pc_in, R, T)

        if self.split == 'train': # Apply random error transform TODO: also for RT2
            max_angle = self.max_r
            rotz = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            roty = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            rotx = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            transl_x = np.random.uniform(-self.max_t, self.max_t)
            transl_y = np.random.uniform(-self.max_t, self.max_t)
            transl_z = np.random.uniform(-self.max_t, self.max_t)

        else:
            initial_RT = self.val_RT[idx]
            rotz = initial_RT[6]
            roty = initial_RT[5]
            rotx = initial_RT[4]
            transl_x = initial_RT[1]
            transl_y = initial_RT[2]
            transl_z = initial_RT[3]

        R = mathutils.Euler((rotx, roty, rotz), 'XYZ') # convert rotation to euler TODO: also RT2
        T = mathutils.Vector((transl_x, transl_y, transl_z)) # convert translation to vector

        # invert R and T because they are the random error we have applied and we want to predcit the transform that can fix the error.
        # So we pass them as the ground truth 
        R, T = invert_pose(R, T) 
        R, T = torch.tensor(R), torch.tensor(T)

        calib = self.K[seq] # K is camera intrinsic transform
        if h_mirror: # hard coded false
            calib[2] = (img.shape[2] / 2)*2 - calib[2]

        if self.split == 'test':
            sample = {
                        'rgb': img, # RGB Image
                        'point_cloud': pc_in, # Lidar Image
                        'point_cloud2': pc2_in,
                        'calib': calib, # Camera Intrinsic 
                        'tr_error': T, 
                        'rot_error': R, 
                        # TODO: 'tr_error2': T2,
                        # TODO: 'rot_error2': R2,
                        'seq': int(seq), 
                        'img_path': img_path,
                        'rgb_name': rgb_name + '.png', 
                        'item': item, 
                        'extrin': RT,
                        'initial_RT': initial_RT
                    }
        else:
            sample = {
                        'rgb': img, 
                        'point_cloud': pc_in, 
                        'point_cloud2': pc2_in,
                        'calib': calib,
                        'tr_error': T, 
                        'rot_error': R, 
                        'seq': int(seq),
                        'rgb_name': rgb_name, 
                        'item': item, 
                        'extrin': RT
                    }

        return sample
    
def main():
    _config = {
        'checkpoints': './checkpoints/',
        'dataset': 'kitti/odom', # 'kitti/raw'
        'data_folder': './data/data_odometry_color/dataset',
        'use_reflectance': False,
        'val_sequence': 0,
        'epochs': 120,
        'BASE_LEARNING_RATE': 3e-4,  # 1e-4,
        'loss': 'combined',
        'max_t': 0.1, # 1.5, 1.0,  0.5,  0.2,  0.1
        'max_r': 1., # 20.0, 10.0, 5.0,  2.0,  1.0
        'batch_size': 32,
        'num_worker': 6,
        'network': 'Res_f1',
        'optimizer': 'adam',
        'resume': True,
        'weights': None, #'./pretrained/kitti_iter5.tar'
        'rescale_rot': 1.0,
        'rescale_transl': 2.0,
        'precision': "O0",
        'norm': 'bn',
        'dropout': 0.0,
        'max_depth': 80.,
        'weight_point_cloud': 0.5,
        'log_frequency': 10,
        'print_frequency': 50,
        'starting_epoch': -1,
    }

    dataset = DatasetLidarCameraKittiOdometry(
        _config['data_folder'], 
        max_r=_config['max_r'], 
        max_t=_config['max_t'],
        split='train', 
        use_reflectance=_config['use_reflectance'],
        val_sequence=_config['val_sequence']
    )

if __name__ == '__main__':
    main()