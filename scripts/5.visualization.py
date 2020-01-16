# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 16:00:40 2020

@author: ning
"""
import os
from glob import glob
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import PIL


output_dir = '../results'
figure_dir = '../figures'
if not os.path.exists(figure_dir):
    os.mkdir(figure_dir)
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
    face = np.load(face_array)
    hand = np.load(hand_array)
    body_array = body_array.replace('\\','/')
    video_name = body_array.split('/')[-2]
    image_name = body_array.split('/')[-1].replace('.npy','.jpeg').replace('frame_','')
    imagePath = os.path.join(output_dir,'frames',video_name,image_name)
    cc = np.array(PIL.Image.open(imagePath))
    figure_saving_dir = os.path.join(figure_dir,video_name)
    if not os.path.exists(figure_saving_dir):
        os.mkdir(figure_saving_dir)
    fig,ax = plt.subplots(figsize = (10,10))
    ax.imshow(cc)
    for ii,person in enumerate(body):
        person = person[person[:,-1] > 0]
        ax.scatter(person[:,0],person[:,1],10,color = 'blue')
        person_face = face[ii,:,:]
        ax.scatter(person_face[:,0],person_face[:,1],10,color='red')
        left_hand = hand[0,ii,:,:]
        right_hand = hand[1,ii,:,:]
        ax.scatter(left_hand[:,0],left_hand[:,1],10,color='yellow')
        ax.scatter(right_hand[:,0],right_hand[:,1],10,color='green')
    fig.savefig(os.path.join(figure_saving_dir,image_name))
    
