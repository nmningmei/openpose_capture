# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 14:41:53 2020

@author: ning
"""

import re
import os
import cv2
import pyopenpose as op
import numpy as np

from glob import glob

face_array_dir = "../results/face_rectangles"
boday_array_dir = "../results/body"
frames_dir = "../results/frames"

params = dict()
params["model_folder"] = "/home/mikel/Downloads/openpose/models/"
params["face"] = True
params["face_detector"] = 2
params["body"] = 0
# params["hand"] = True
# params["hand_detector"] = 2

frame_folders = os.listdir(frames_dir)

allImagePaths = np.sort(glob(os.path.join(frames_dir,'*','*.jpeg')))

idx = 0 # batch change

imagePath = allImagePaths[idx]
imagePath = imagePath.replace('\\','/')
frame_folder = imagePath.split('/')[-2]