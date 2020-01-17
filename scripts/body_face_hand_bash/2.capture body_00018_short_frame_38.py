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
import numpy      as np

from glob import glob

# the rest of the parameters are left as default
params                  = dict()
params["model_folder"]  = "/home/mikel/Downloads/openpose/models/"
frames_dir              = "../../results/frames"
body_dir                = '../../results/body'
face_rect_dir           = '../../results/face_rectangles'
hand_rect_dir           = "../../results/hand_rectangles"
for d in [body_dir,face_rect_dir,hand_rect_dir]:
    if not os.path.exists(d):
        os.mkdir(d)

allImagePaths           = np.sort(glob(os.path.join(frames_dir,'*','*.jpeg')))

idx = 85 # batch change

imagePath               = allImagePaths[idx]
imagePath               = imagePath.replace('\\','/')
frame_folder            = imagePath.split('/')[-2]
frame_index             = re.findall("\d+",imagePath)[-1]

# Starting OpenPose
opWrapper               = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()
datum                   = op.Datum()
imageToProcess          = cv2.imread(imagePath)

datum.cvInputData       = imageToProcess
opWrapper.emplaceAndPop([datum])
print(datum.poseKeypoints)

# save the body key points
array_dir               = os.path.join(body_dir,frame_folder)
if not os.path.exists(array_dir):
    os.mkdir(array_dir)
saving_name             = f'frame_{frame_index}'
np.save(os.path.join(array_dir,
                     f"{saving_name}.npy"),
        datum.poseKeypoints)

# save face rectangles
face_array_dir          = os.path.join(face_rect_dir,frame_folder)
hand_array_dir          = os.path.join(hand_rect_dir,frame_folder)
for d in [face_array_dir,hand_array_dir]:
    if not os.path.exists(d):
        os.mkdir(d)
face_rectangles         = []
hand_rectangles         = []
for person in datum.poseKeypoints: # in case multiple people detected
    leftear             = person[17]
    rightear            = person[18]
    nose                = person[0]
    lefthand            = person[7]
    righthand           = person[4]
    if not leftear[-1] > 0: # if not detecting left ear
        target_ear      = rightear.copy()
    else:
        target_ear      = leftear.copy()
    # the length of the sides of the square is defined 
    # by the distance between the nose and one of the ear
    # times a constant
    w                   = 2 * np.sqrt(((target_ear[0]-nose[0])**2 +\
                                       (target_ear[1]-nose[1])**2))
    # make it even bigger
    w                   *= 1.5
    h                   = w
    # the bottom left corner
    x,y                 = nose[0] - w/2, nose[1] - w/2
    face_rectangles.append([x,y,h,w])
    
    # rescale the sqaure for hand
    w                   = w / 1.5 * 2
    h                   = w
    for hand,hand_name in zip([lefthand,righthand],
                              ["lefthand","righthand"]):
        if hand[-1] > 0:
            x,y             = hand[0] - w/2, hand[1] - w/2
            hand_retangle   = np.array([x,y,h,w])
            hand_rectangles.append([x,y,h,w])
face_rectangles         = np.array(face_rectangles)
hand_rectangles         = np.array(hand_rectangles)
np.save(os.path.join(face_array_dir,
                     f"{saving_name}.npy"),
        face_rectangles)
np.save(os.path.join(hand_array_dir,
                     f"{saving_name}.npy"),
        hand_rectangles)



#from matplotlib import pyplot as plt
#from matplotlib.patches import Rectangle
#import PIL
#
#cc = np.array(PIL.Image.open(imagePath))
#fig,ax = plt.subplots(figsize = (10,10))
#ax.imshow(cc)
#for people in data:
#    people = people[people[:,-1] > 0]
#    ax.scatter(people[:,0],people[:,1],10,)
#for a in face_rectangles:
#    x,y,w,h = a
#    rect_ = Rectangle((x,y), w, h,
#              linewidth = 1,
#              edgecolor = "red",
#              facecolor = None,
#              fill = False,)
#    ax.add_patch(rect_)
#for a in hands:
#    x,y,w,h = a
#    rect_ = Rectangle((x,y), w, h,
#              linewidth = 1,
#              edgecolor = "red",
#              facecolor = None,
#              fill = False,)
#    ax.add_patch(rect_)









