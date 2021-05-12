#!/usr/bin/env python3


import os, sys

import matplotlib.pyplot as plt
import numpy as np
import cv2

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
  # Data obtained from the SunRGBD Dataset

  # 3: bed detected and 189: no object
  img_name = f'../../data/img-000003.jpg'
  img_path = os.path.join(sys.path[0], img_name)
  rgb_img = plt.imread(img_path)

  depth_name = f'../../data/3.png'
  depth_path = os.path.join(sys.path[0], depth_name)
  depth_img = plt.imread(depth_path)


  point3d = convert_depth_to_pointcloud(depth_img)
  net = votenet_build()
  detections = votenet_forward(net, point3d)

  
  frame_results = []
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
      frame_results.append(result)
  print(frame_results)


if __name__ == '__main__':
  print("\n\nWelcome\n\n")
  readImages()
  print("\n\nExiting\n\n")