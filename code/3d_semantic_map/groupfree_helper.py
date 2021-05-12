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

from scipy.spatial.transform import Rotation

import Group
from Group import models
from models import GroupFreeDetector, get_loss

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Group')
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from models import ap_helper
from ap_helper import parse_predictions
from pc_util import random_sampling
from nms import nms_2d_faster, nms_3d_faster, nms_3d_faster_samecls

# Default settings
DATASET = 'sunrgbd'  # supported values: sunrgbd, scannet
NUM_POINTS = 40000  # number of points used in sampling the point cloud


def cleanPointCloud(point_cloud):
    point_cloud = point_cloud[:, 0:3]  # do not use color for now
    floor_height = np.percentile(point_cloud[:, 2], 0.99)
    height = point_cloud[:, 2] - floor_height
    # point_cloud = np.concatenate(
        # [point_cloud, np.expand_dims(height, 1)], 1)  # (N,4) or (N,7)
    point_cloud = random_sampling(point_cloud, NUM_POINTS)
    pc = np.expand_dims(point_cloud.astype(np.float32), 0)  # (1,40000,4)
    return pc

def groupfree_build():
    # Set file paths and dataset config
    demo_dir = os.path.join(ROOT_DIR, 'demo_files')
    print(demo_dir)
    if DATASET == 'sunrgbd':
        print("hello")
        sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
        from sunrgbd_detection_dataset import DC  # dataset config

        checkpoint_path = os.path.join(demo_dir,
                                       'sunrgbd_l6o256_cls_agnostic.pth')
        print(checkpoint_path)
        # pc_path = os.path.join(demo_dir, 'input_pc_sunrgbd.ply')
    else:
        print('Unkown dataset %s. Exiting.' % (DATASET))
        exit(-1)

    eval_config_dict = {
        'remove_empty_box': True,
        'use_3d_nms': True,
        'nms_iou': 0.25,
        'use_old_type_nms': True,
        'cls_nms': True,
        'per_class_proposal': False,
        'conf_thresh': 0.6,
        'dataset_config': DC
    }

    # Init the model and optimzier
    MODEL = importlib.import_module('Group')  # import network module
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    from sunrgbd.model_util_sunrgbd import SunrgbdDatasetConfig
    DATASET_CONFIG = SunrgbdDatasetConfig()
    net = GroupFreeDetector(num_class=DATASET_CONFIG.num_class,
                              num_heading_bin=DATASET_CONFIG.num_heading_bin,
                              num_size_cluster=DATASET_CONFIG.num_size_cluster,
                              mean_size_arr=DATASET_CONFIG.mean_size_arr,
                              input_feature_dim=0,
                              width=1,
                              num_proposal=256,
                              sampling='kps',
                              dropout=0.1,
                              activation='relu',
                              nhead=8,
                              num_decoder_layers=6,
                              dim_feedforward=2048,
                              self_position_embedding='loc_learned',
                              cross_position_embedding='xyz_learned',
                              size_cls_agnostic=True)
    print('Constructed model.')

    # Load checkpoint
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    checkpoint = torch.load(checkpoint_path,map_location='cpu')
    state_dict = checkpoint['model']
    save_path = checkpoint.get('save_path', 'none')
    for k in list(state_dict.keys()):
        state_dict[k[len("module."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]
    net.load_state_dict(state_dict)
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # epoch = checkpoint['epoch']
    # print("Loaded checkpoint %s (epoch: %d)" % (checkpoint_path, epoch))
    print(device)
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


def groupfree_forward(net, point_cloud):
    net.eval()
    pc = cleanPointCloud(point_cloud)

    inputs = {'point_clouds': torch.from_numpy(pc).to(net.device)}
    end_points = None
    with torch.no_grad():
        end_points = net(inputs)
    end_points['point_clouds'] = inputs['point_clouds']
    # print(end_points['point_clouds'].shape)

    _prefixes = ['last_', 'proposal_']
    _prefixes += [f'{i}head_' for i in range(5)]
    # prefixes = _prefixes.copy() + ['all_layers_']

    # ap_calculator_list = [APCalculator(iou_thresh, DATASET_CONFIG.class2type) \
                          # for iou_thresh in AP_IOU_THRESHOLDS]

    # batch_pred_map_cls_dict = {k: [] for k in prefixes}
    # batch_gt_map_cls_dict = {k: [] for k in prefixes}
    prefix = 'all_layers_'

    end_points[f'{prefix}center'] = torch.cat([end_points[f'{ppx}center']
                                               for ppx in _prefixes], 1)
    end_points[f'{prefix}heading_scores'] = torch.cat([end_points[f'{ppx}heading_scores']
                                                       for ppx in _prefixes], 1)
    end_points[f'{prefix}heading_residuals'] = torch.cat([end_points[f'{ppx}heading_residuals']
                                                                      for ppx in _prefixes], 1)
    end_points[f'{prefix}pred_size'] = torch.cat([end_points[f'{ppx}pred_size']
                                                                  for ppx in _prefixes], 1)

    end_points[f'{prefix}sem_cls_scores'] = torch.cat([end_points[f'{ppx}sem_cls_scores']
                                                                   for ppx in _prefixes], 1)
    end_points[f'{prefix}objectness_scores'] = torch.cat([end_points[f'{ppx}objectness_scores']
                                                                      for ppx in _prefixes], 1)



    

    pred_map_cls = parse_predictions(end_points, net.eval_config_dict,prefix)
    print(f'{len(pred_map_cls[0])} object(s) detected')
    return pred_map_cls

def compute_world_T_camera(pose_rotation, pose_translation):
    """
    This function computes the transformation matrix T that transforms one point in the camera coordinate system to
    the world coordinate system. This function needs the scipy package - scipy.spatial.transform.Rotation
    P_w = P_r * T, where P_r = [x, y, z] is a row vector in the camera coordinate (x - right, y - down, z - forward)
    and P_w the location of the point in the world coordinate system (x - right, y - forward, z - up)

    :param pose_rotation: [quat_x, quat_y, quat_z, quat_w], a quaternion that represents a rotation
    :param pose_translation: [tr_x, tr_y, tr_z], the location of the robot origin in the world coordinate system
    :return: transformation matrix
    """
    # compute the rotation matrix
    # note that the scipy...Rotation accepts the quaternion in scalar-last format
    pose_quat = pose_rotation
    rot = Rotation.from_quat(pose_quat)
    rotation_matrix = np.eye(4)
    rotation_matrix[0:3][:, 0:3] = rot.as_matrix().T

    # build the tranlation matrix
    translation_matrix = np.eye(4)
    translation_matrix[3][0:3] = np.array(pose_translation)

    # compute the tranformation matrix
    transformation_matrix = rotation_matrix @ translation_matrix

    # # convert robot's right-handed coordinate system to unreal world's left-handed coordinate system
    # transformation_matrix[:, 1] = -1.0 * transformation_matrix[:, 1]

    return transformation_matrix

def convert_depth_to_pointcloud(observations):
    # depth_image = observations['image_depth']
    depth_image = observations
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


def groupfree_detection(net, observations):
    # build the transformation matrix w_T_c that transforms
    # the point in the camera coordinate system into the world system
    # P_w = P_c * w_T_c, P_c = [x, y, z, 1] a row vector denoting a point location in the camera coordinate
    camera_pose = observations['poses']['camera']
    w_T_c = compute_world_T_camera(camera_pose['rotation_xyzw'],
                                   camera_pose['translation_xyz'])
    # print(np.array([0, 0, 1, 1], dtype=np.float) @ w_T_c)

    point3d = convert_depth_to_pointcloud(observations)

    # forward the point cloud into the groupfree trained on SUNRGBD dataset
    detections = groupfree_forward(net, point3d)

    # parse the detection results for current frame
    frame_results = []
    print("\n>>>>>>> Printing detections:")
    if detections[0]:
        detections = detections[0]
        for detection in detections:
            # ignore those classes that are not in ACRV classes
            if detection[0] > net.type2class['desk']:
                continue
            result = {
                "class": net.class2type[detection[0]],
                "class_ID": net.type2class[net.class2type[detection[0]]],
                "confidence": np.float64(detection[2]),
                "box_corners_camera": detection[1]
            }

            # convert the 3D box corners in the camera coordinate system to the world coordinate system
            box_corners_camera = detection[1]
            box_corners_camera = np.concatenate(
                (box_corners_camera,
                 np.ones((box_corners_camera.shape[0], 1),
                         dtype=box_corners_camera.dtype)),
                axis=1)
            box_corners_world = box_corners_camera @ w_T_c
            box_corners_world = np.array(box_corners_world[:, 0:3])
            # Normalizing floor height in the z-coordinate
            box_corners_world[:,2] -= 1.5
            result["box_corners_world"] = box_corners_world
            # compute the 3D box's center roughly
            box_center_world = np.sum(box_corners_world, axis=0) / 8.0
            result["centroid"] = box_center_world
            # compute the 3D box's extent roughly
            box_extent_world = np.sum(
                np.abs(box_corners_world - box_center_world), axis=0) / 4.0
            result["extent"] = box_extent_world

            print(net.class2type[detection[0]], [round(i,2) for i in result["centroid"]])
            frame_results.append(result)

    print("<<<<<< Finished printing detections!")
    return frame_results


def groupfree_nms(all_results, net, class_list):
    nms_iou_cls = 0.1  # this threshold for removing duplicates that are of the same class
    boxes_world = []
    results_list = []
    for results in all_results:
        if not results:
            continue
        for result in results:
            box_world = np.zeros((1, 8), dtype=np.float32)
            box_corners_world = result["box_corners_world"]
            box_world[0, 0] = np.min(box_corners_world[:, 0])
            box_world[0, 1] = np.min(box_corners_world[:, 1])
            box_world[0, 2] = np.min(box_corners_world[:, 2])
            box_world[0, 3] = np.max(box_corners_world[:, 0])
            box_world[0, 4] = np.max(box_corners_world[:, 1])
            box_world[0, 5] = np.max(box_corners_world[:, 2])
            box_world[0, 6] = result["confidence"]
            box_world[0, 7] = result["class_ID"]
            boxes_world.append(box_world)
            results_list.append(result)
    boxes_world = np.concatenate(boxes_world)

    pick = nms_3d_faster_samecls(boxes_world, nms_iou_cls,
                                 net.eval_config_dict['use_old_type_nms'])
    assert (len(pick) > 0)
    results_intermediate = [results_list[p] for p in pick]

    nms_iou = 0.25  # this threshold is used to remove duplicates that are not of the same class
    boxes_world = []
    results_list = []
    for result in results_intermediate:
        if not result:
            continue
        box_world = np.zeros((1, 7), dtype=np.float32)
        box_corners_world = result["box_corners_world"]
        box_world[0, 0] = np.min(box_corners_world[:, 0])
        box_world[0, 1] = np.min(box_corners_world[:, 1])
        box_world[0, 2] = np.min(box_corners_world[:, 2])
        box_world[0, 3] = np.max(box_corners_world[:, 0])
        box_world[0, 4] = np.max(box_corners_world[:, 1])
        box_world[0, 5] = np.max(box_corners_world[:, 2])
        box_world[0, 6] = result["confidence"]
        # box_world[0, 7] = result["class_ID"]
        boxes_world.append(box_world)
        results_list.append(result)
    boxes_world = np.concatenate(boxes_world)

    pick = nms_3d_faster(boxes_world, nms_iou,
                         net.eval_config_dict['use_old_type_nms'])
    assert (len(pick) > 0)

    results_final = [results_list[p] for p in pick]
    results_final = jsonify(results_final)

    for r in results_final:
        r['label_probs'] = [0] * len(class_list)
        if r['class'] in class_list:
            r['label_probs'][class_list.index(r['class'])] = r['confidence']

    return results_final


def jsonify(data_list):
    json_list = []
    for data in data_list:
        json_data = dict()
        for key, value in data.items():
            if isinstance(value, list):  # for lists
                value = [
                    jsonify(item) if isinstance(item, dict) else item
                    for item in value
                ]
            if isinstance(value, dict):  # for nested lists
                value = jsonify(value)
            if isinstance(key, int):  # if key is integer: > to string
                key = str(key)
            if type(
                    value
            ).__module__ == 'numpy':  # if value is numpy.*: > to python list
                value = value.tolist()
            json_data[key] = value
        json_list.append(json_data)
    return json_list
