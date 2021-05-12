# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import numpy as np
import importlib
import time
import torch
import torch.nn as nn
import torch.optim as optim


ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/votenet')
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from ap_helper import parse_predictions
from pc_util import random_sampling

# Default settings
DATASET = 'sunrgbd'  # supported values: sunrgbd, scannet
NUM_POINTS = 40000  # number of points used in sampling the point cloud


def cleanPointCloud(point_cloud):
    point_cloud = point_cloud[:, 0:3]  # do not use color for now
    floor_height = np.percentile(point_cloud[:, 2], 0.99)
    height = point_cloud[:, 2] - floor_height
    point_cloud = np.concatenate(
        [point_cloud, np.expand_dims(height, 1)], 1)  # (N,4) or (N,7)
    point_cloud = random_sampling(point_cloud, NUM_POINTS)
    pc = np.expand_dims(point_cloud.astype(np.float32), 0)  # (1,40000,4)
    return pc


def votenet_build():
    # Set file paths and dataset config
    demo_dir = os.path.join(ROOT_DIR, 'demo_files')
    if DATASET == 'sunrgbd':
        sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
        from sunrgbd_detection_dataset_0 import DC  # dataset config

        checkpoint_path = os.path.join(demo_dir,
                                       'pretrained_votenet_on_sunrgbd.tar')
        pc_path = os.path.join(demo_dir, 'input_pc_sunrgbd.ply')
    elif DATASET == 'scannet':
        sys.path.append(os.path.join(ROOT_DIR, 'scannet'))
        from scannet_detection_dataset import DC  # dataset config

        checkpoint_path = os.path.join(demo_dir,
                                       'pretrained_votenet_on_scannet.tar')
        pc_path = os.path.join(demo_dir, 'input_pc_scannet.ply')
    else:
        print('Unkown dataset %s. Exiting.' % (DATASET))
        exit(-1)

    eval_config_dict = {
        'remove_empty_box': True,
        'use_3d_nms': True,
        'nms_iou': 0.25,
        'use_old_type_nms': False,
        'cls_nms': False,
        'per_class_proposal': False,
        'conf_thresh': 0.5,
        'dataset_config': DC
    }

    # Init the model and optimzier
    MODEL = importlib.import_module('votenet')  # import network module
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = MODEL.VoteNet(num_proposal=256,
                        input_feature_dim=1,
                        vote_factor=1,
                        sampling='seed_fps',
                        num_class=DC.num_class,
                        num_heading_bin=DC.num_heading_bin,
                        num_size_cluster=DC.num_size_cluster,
                        mean_size_arr=DC.mean_size_arr).to(device)
    print('Constructed model.')

    # Load checkpoint
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print("Loaded checkpoint %s (epoch: %d)" % (checkpoint_path, epoch))

    net.device = device
    net.eval_config_dict = eval_config_dict

    type2class = {
        'bed': 0,
        'table': 1,
        'sofa': 2,
        'chair': 3,
        'toilet': 4,
        'desk': 5,
        'dresser': 6,
        'night_stand': 7,
        'bookshelf': 8,
        'bathtub': 9
    }
    class2type = {type2class[t]: t for t in type2class}
    # merge 'table:1' and 'desk:5' into 'table'
    class2type[5] = 'table'
    net.type2class = type2class
    net.class2type = class2type

    return net


def votenet_forward(net, point_cloud):
    net.eval()
    pc = cleanPointCloud(point_cloud)

    inputs = {'point_clouds': torch.from_numpy(pc).to(net.device)}
    end_points = None
    with torch.no_grad():
        end_points = net(inputs)
    end_points['point_clouds'] = inputs['point_clouds']
    pred_map_cls = parse_predictions(end_points, net.eval_config_dict)
    print(f'{len(pred_map_cls[0])} object(s) detected')
    return pred_map_cls


# def convert_depth_to_pointcloud(depth_image):
#     image_size = depth_image.shape
#     nx = np.arange(0, image_size[1])
#     ny = np.arange(0, image_size[0])
#     x, y = np.meshgrid(nx, ny)
#     f = np.array([480], dtype=np.float32)
#     cx = image_size[1] / 2.0
#     cy = image_size[0] / 2.0
#     x3 = (x - cx) * depth_image * 1 / f
#     y3 = (y - cy) * depth_image * 1 / f
#     point3d = np.stack((x3.flatten(), depth_image.flatten(), -y3.flatten()),
#                        axis=1)

#     return point3d

def convert_depth_to_pointcloud(depth_image):
    image_size = depth_image.shape
    nx = np.zeros(image_size[1])
    ny = np.zeros(image_size[0])
    for i in range(image_size[1]):
        nx[i] = i
    for i in range(image_size[0]):
        nx[i] = i
    x, y = np.meshgrid(nx, ny)
    f = np.array([480], dtype=np.float32)
    cx = image_size[1] / 2.0
    cy = image_size[0] / 2.0
    x3 = (x - cx) * depth_image * 1 / f
    y3 = (y - cy) * depth_image * 1 / f
    point3d = np.stack((x3.flatten(), depth_image.flatten(), -y3.flatten()),
                       axis=1)

    return point3d