#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 12:06:42 2020

@author: mikel
"""


import cv2
import re
import numpy as np

weights = "../model/pose_iter_102000.caffemodel"
protofile = "../model/pose_deploy.prototxt"
imagePath = "../data/frame2137.jpg"
image_name = re.findall("frame\d+",imagePath)[0]
nPoints = 22
imageforprocessing = cv2.imread(imagePath)
net = cv2.dnn.readNetFromCaffe(protofile,weights)

aspect_ratio = imageforprocessing.shape[0]/imageforprocessing.shape[1]
inHeight = 368
inWidth = int(((aspect_ratio*inHeight)*8)//8)
inpBlob = cv2.dnn.blobFromImage(
    imageforprocessing, 1.0 / 255, 
    (inWidth, inHeight),
    (0, 0, 0), 
    swapRB=False, 
    crop=False)
 
net.setInput(inpBlob)
 
output = net.forward()

points = []
 
for i in range(nPoints):
    # confidence map of corresponding body's part.
    probMap = output[0, i, :, :]
    frameWidth, frameHeight = imageforprocessing.shape[0],imageforprocessing.shape[1]
    probMap = cv2.resize(probMap, (frameWidth, frameHeight))
 
    # Find global maxima of the probMap.
    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
 
    if prob > .1 :
        cv2.circle(imageforprocessing, (int(point[0]), int(point[1])), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
        cv2.putText(imageforprocessing, "{}".format(i), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)
 
        # Add the point to the list if the probability is greater than the threshold
        points.append((int(point[0]), int(point[1])))
    
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import PIL
points = np.array(points)
cc = np.array(PIL.Image.open(imagePath))
fig,ax = plt.subplots(figsize = (10,10))
ax.imshow(cc)
ax.scatter(points[:,0],
           points[:,1],
           s = 10)
