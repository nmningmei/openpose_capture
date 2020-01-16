#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 13:37:44 2019

@author: mikel
"""

import os
import re
import cv2
import pyopenpose as op
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import PIL

import os
from glob import glob
from tqdm import tqdm

params = dict()
params["model_folder"] = "../../openpose/models/"
params["face"] = True
params["face_detector"] = 2
params["body"] = 0
# params["hand"] = True
# params["hand_detector"] = 2

imagePath = "../data/frame2137.jpg"
image_name = re.findall("frame\d+",imagePath)[0]
face_array_dir = "../results/face_rectangles"
boday_array_dir = "../results/body"

face_rects = np.load(os.path.join(face_array_dir,
                                  f"{image_name}.npy"))
# Starting OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()
datum = op.Datum()
imageToProcess = cv2.imread(imagePath)
faceRectangles = []
for a in face_rects:
    faceRectangles.append(op.Rectangle(*list(a)))
        
datum.cvInputData = imageToProcess
datum.faceRectangles = faceRectangles
opWrapper.emplaceAndPop([datum])
print("Face keypoints: \n" + str(datum.faceKeypoints))
array_dir = "../results/face"
if not os.path.exists(array_dir):
    os.mkdir(array_dir)
data = datum.faceKeypoints
saving_name = re.findall("frame\d+",imagePath)[0]
np.save(os.path.join(array_dir,
                     f"{saving_name}.npy"),
        data)
# cc = np.array(PIL.Image.open(imagePath))
# body_points = np.load(os.path.join(
#     boday_array_dir,
#     f"{image_name}.npy"))
# fig,ax = plt.subplots(figsize = (10,10))
# ax.imshow(cc)
# for a in face_rects:
#     (x, y, w, h) = a
#     rect_ = Rectangle((x,y), w, h,
#               linewidth = 1,
#               edgecolor = "red",
#               facecolor = None,
#               fill = False,)
#     ax.add_patch(rect_)
# for data in datum.faceKeypoints:
#     data = data[data[:,-1] > 0]
#     ax.scatter(data[:,0],
#                 data[:,1],
#                 10,)
# for data in body_points:
#     data = data[data[:,-1] > 0]
#     ax.scatter(data[:,0],
#                 data[:,1],
                10,)






