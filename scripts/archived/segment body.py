#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 13:37:44 2019

@author: mikel
"""

import re
import os
import cv2
import pyopenpose as op
import numpy as np

params = dict()
params["model_folder"] = "/home/mikel/Downloads/openpose/models"

imagePath = "../results/frames/00018_short/116.jpeg"

# Starting OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()
datum = op.Datum()
imageToProcess = cv2.imread(imagePath)

datum.cvInputData = imageToProcess
opWrapper.emplaceAndPop([datum])
print(datum.poseKeypoints)

array_dir = "../results/body"
if not os.path.exists(array_dir):
    os.mkdir(array_dir)
data = datum.poseKeypoints
saving_name = "temp"#re.findall("frame\d+",imagePath)[0]
np.save(os.path.join(array_dir,
                     f"{saving_name}.npy"),
        data)

array_dir = "../results/face_rectangles"
if not os.path.exists(array_dir):
    os.mkdir(array_dir)
face_rectangles = []
for person in data:
    leftear = person[17]
    rightear = person[18]
    nose = person[0]
    if not leftear[-1]>0:
        target_ear = rightear.copy()
    else:
        target_ear = leftear.copy()
    w = 2 *np.sqrt(((target_ear[0]-nose[0])**2 +\
                    (target_ear[1]-nose[1])**2))
    w *= 1.5
    h = w
    x,y = nose[0] - w/2, nose[1] - w/2
    face_rectangles.append([x,y,h,w])
face_rectangles = np.array(face_rectangles)
np.save(os.path.join(array_dir,
                     f"{saving_name}.npy"),
        face_rectangles)

array_dir = "../results/hand_rectangles"
if not os.path.exists(array_dir):
    os.mkdir(array_dir)
hands = []
for person in data:
    lefthand = person[7]
    righthand = person[4]
    leftear = person[17]
    rightear = person[18]
    nose = person[0]
    if not leftear[-1]>0:
        target_ear = rightear.copy()
    else:
        target_ear = leftear.copy()
    w = 2 *np.sqrt(((target_ear[0]-nose[0])**2 +\
                    (target_ear[1]-nose[1])**2))
    w *= 2
    h = w
    for hand,hand_name in zip([lefthand,righthand],
                               ["lefthand","righthand"]):
        if hand[-1] > 0:
            x,y = hand[0] - w/2, hand[1] - w/2
            hand_retangle = np.array([x,y,h,w])
            
            hands.append([x,y,h,w])
hands = np.array(hands)
np.save(os.path.join(array_dir,
                f"{saving_name}.npy"),
                hands)

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import PIL

cc = np.array(PIL.Image.open(imagePath))
fig,ax = plt.subplots(figsize = (10,10))
ax.imshow(cc)
for a in face_rectangles:
    x,y,w,h = a
    rect_ = Rectangle((x,y), w, h,
              linewidth = 1,
              edgecolor = "red",
              facecolor = None,
              fill = False,)
    ax.add_patch(rect_)
for a in hands:
    x,y,w,h = a
    rect_ = Rectangle((x,y), w, h,
              linewidth = 1,
              edgecolor = "red",
              facecolor = None,
              fill = False,)
    ax.add_patch(rect_)









