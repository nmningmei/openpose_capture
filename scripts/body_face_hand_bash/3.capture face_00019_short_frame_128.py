# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 14:41:53 2020

@author: ning
"""

import re
import os
import cv2
import pyopenpose as op
import numpy      as np

from glob import glob

face_array_dir          = "../../results/face_rectangles"
boday_array_dir         = "../../results/body"
face_array_saving_dir   = '../../results/face/'
frames_dir              = "../../results/frames"
if not os.path.exists(face_array_saving_dir):
    os.mkdir(face_array_saving_dir)

params                  = dict()
params["model_folder"]  = "/home/mikel/Downloads/openpose/models/"
params["face"]          = True
params["face_detector"] = 2
params["body"]          = 0

allImagePaths           = np.sort(glob(os.path.join(frames_dir,
                                                    '*',
                                                    '*.jpeg')))

idx                     = 185 # batch change

imagePath               = allImagePaths[idx]
imagePath               = imagePath.replace('\\','/')
frame_folder            = imagePath.split('/')[-2]
frame_index             = re.findall("\d+",imagePath)[-1]

face_rects              = np.load(os.path.join(face_array_dir,
                                               frame_folder,
                                               f"frame_{frame_index}.npy"))
# Starting OpenPose
opWrapper               = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()
datum                   = op.Datum()
imageToProcess          = cv2.imread(imagePath)
faceRectangles          = []
for a in face_rects:
    faceRectangles.append(op.Rectangle(*list(a)))
# define the image input and the face rectangles
datum.cvInputData       = imageToProcess
datum.faceRectangles    = faceRectangles
opWrapper.emplaceAndPop([datum])
print("Face keypoints: \n" + str(datum.faceKeypoints))

array_dir               = os.path.join(face_array_saving_dir,frame_folder)
if not os.path.exists(array_dir):
    os.mkdir(array_dir)
saving_name             = f'frame_{frame_index}'
np.save(os.path.join(array_dir,
                     f"{saving_name}.npy"),
        datum.faceKeypoints)













