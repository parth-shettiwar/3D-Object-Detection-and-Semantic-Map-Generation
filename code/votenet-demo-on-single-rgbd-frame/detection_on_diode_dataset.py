#!/usr/bin/env python3


import os, sys

import matplotlib.pyplot as plt
import numpy as np
import cv2

# import open3d
# from open3d import *

from helper import convert_depth_to_pointcloud, votenet_build, votenet_forward

def plot_depth_map(dm, validity_mask):
  validity_mask = validity_mask > 0
  MIN_DEPTH = 0.5
  MAX_DEPTH = min(300, np.percentile(dm, 99))
  dm = np.clip(dm, MIN_DEPTH, MAX_DEPTH)
  dm = np.log(dm, where=validity_mask)

  dm = np.ma.masked_where(~validity_mask, dm)

  cmap = plt.cm.get_cmap("jet").copy()
  cmap.set_bad(color='black')
  plt.imshow(dm, cmap=cmap, vmax=np.log(MAX_DEPTH))

def displayRGBDImages(rgb_img, depth_img, validity_mask):  
  plt.subplot(211)
  plt.imshow(rgb_img)

  plt.subplot(212)
  plot_depth_map(depth_img, validity_mask)
  plt.show()

def readImages():
  # Data obtained from the DIODE(Dense Indoor and Outdoor DEpth) Dataset
  img_name = '../../data/00019_00183_indoors_150_000'
  img_path = os.path.join(sys.path[0], f'{img_name}.png')
  rgb_img = plt.imread(img_path)

  depth_path = os.path.join(sys.path[0], f'{img_name}_depth.npy')
  depth_validity_path = os.path.join(sys.path[0], f'{img_name}_depth_mask.npy')
  depth_img = np.load(depth_path)
  depth_img = depth_img.reshape(depth_img.shape[:2])
  validity_mask = np.load(depth_validity_path)

  displayRGBDImages(rgb_img, depth_img, validity_mask)
  point3d = convert_depth_to_pointcloud(depth_img)
  net = votenet_build()
  detections = votenet_forward(net, point3d)
  # rgbd = {}
  # rgbd['color'] = rgb_img
  # rgbd['depth'] = depth_img
  # open3d.geometry.create_point_cloud_from_rgbd_image(rgbd, pinhole_camera_intrinsic)


if __name__ == '__main__':
  print("\n\nWelcome\n\n")
  readImages()
  print("\n\nExiting\n\n")