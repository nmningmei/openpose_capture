#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 15:28:44 2020

@author: mikel
"""

import re
import os
import cv2
import pyopenpose as op
import numpy as np

from glob import glob

hand_array_dir = "../../results/hand_rectangles"
boday_array_dir = "../../results/body"
frames_dir = "../../results/frames"

params = dict()
params["model_folder"] = "/home/mikel/Downloads/openpose/models/"
params["hand"] = True
params["hand_detector"] = 2
params["body"] = 0

allImagePaths = np.sort(glob(os.path.join(frames_dir,'*','*.jpeg')))

idx = 254 # batch change

imagePath = allImagePaths[idx]
imagePath = imagePath.replace('\\','/')
frame_folder = imagePath.split('/')[-2]
frame_index = re.findall("\d+",imagePath)[-1]

hand_rects = np.load(os.path.join(hand_array_dir,frame_folder,f"frame_{frame_index}.npy"))
data = []
for hand_corr in hand_rects:
    hand_corr
    handRectangles = [[op.Rectangle(*list(hand_corr)),
                       op.Rectangle(0.,0.,0.,0.)]]
    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()
    datum = op.Datum()
    imageToProcess = cv2.imread(imagePath)
    datum.cvInputData = imageToProcess
    datum.handRectangles = handRectangles
    opWrapper.emplaceAndPop([datum])
    print("Hand keypoints: \n" + str(datum.handKeypoints))
    data.append(datum.handKeypoints[0])
data = np.array(data)

array_dir = os.path.join("../../results/hand",frame_folder)
if not os.path.exists(array_dir):
    os.makedirs(array_dir)
data = datum.faceKeypoints
saving_name = f'frame_{frame_index}'
np.save(os.path.join(array_dir,
                     f"{saving_name}.npy"),
        data)

#from matplotlib import pyplot as plt
#from matplotlib.patches import Rectangle
#import PIL
#
#cc = np.array(PIL.Image.open(imagePath))
#fig,ax = plt.subplots(figsize = (10,10))
#ax.imshow(cc)
#for a in hand_rects:
#    x,y,w,h = a
#    rect_ = Rectangle((x,y), w, h,
#              linewidth = 1,
#              edgecolor = "red",
#              facecolor = None,
#              fill = False,)
#    ax.add_patch(rect_)
#for hand in data:
#    hand = hand[0]
#    ax.scatter(hand[:,0],hand[:,1],10)







