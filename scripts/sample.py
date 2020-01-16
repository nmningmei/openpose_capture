#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 13:37:44 2019

@author: mikel
"""


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
params["model_folder"] = "/home/mikel/Downloads/openpose/models/"
# params["face"] = True
# params["face_detector"] = 2
# params["body"] = 0
# params["hand"] = True
# params["hand_detector"] = 2

imagePath = "/home/mikel/Downloads/frame2137.jpg"

# construct the argument parser and parse the arguments
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

image = cv2.imread(imagePath)
# image = cv2.resize(image, (500,250))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# detect faces in the grayscale image
rects = detector(gray, 1) 
face_rects = []
for a in rects:
    (x, y, w, h) = face_utils.rect_to_bb(a)
    face_rects.append([x,y,w,w])
face_rects = np.array(face_rects)
# Starting OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()
datum = op.Datum()
imageToProcess = cv2.imread(imagePath)
# imageToProcess = cv2.resize(imageToProcess, (500,250))
faceRectangles = []
for a in face_rects:
    a = a.astype(int)
    faceRectangles.append(op.Rectangle(*list(a)))
        
datum.cvInputData = imageToProcess
# datum.faceRectangles = faceRectangles
opWrapper.emplaceAndPop([datum])
# print("Face keypoints: \n" + str(datum.faceKeypoints))
# print(datum.handKeypoints)


cc = np.array(PIL.Image.open(imagePath))
fig,ax = plt.subplots(figsize = (10,10))
ax.imshow(cc)
# for a in rects:
#     (x, y, w, h) = face_utils.rect_to_bb(a)
#     rect_ = Rectangle((x,y), w, h,
#               linewidth = 1,
#               edgecolor = "red",
#               facecolor = None,
#               fill = False,)
#     ax.add_patch(rect_)
# for data in datum.faceKeypoints:
#     data = data[data[:,-1] > 0]
#     ax.scatter(data[:,0],
#                data[:,1],
#                10,)
for data in datum.poseKeypoints:
    data = data[data[:,-1] > 0]
    ax.scatter(data[:,0],
                data[:,1],
                50,)






