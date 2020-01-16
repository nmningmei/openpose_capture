# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 16:00:40 2020

@author: ning
"""
import os
from glob import glob
import numpy as np
output_dir = '../results'
directories = {}
for name in ['body','face','hand']:
    temp = os.path.join(output_dir,name)
    directories[name] = temp.replace('\\','/')

array_files = {}
for name in ['body','face','hand']:
    temp = np.sort(glob(os.path.join(directories[name],'*','*.npy')))
    array_files[name] = temp

collections = np.array(list(array_files.values()))

for (body_array,face_array,hand_array) in collections.T:
    body = np.load(body_array)
    face = np.load(hand_array)
    hand = np.load(hand_array)
