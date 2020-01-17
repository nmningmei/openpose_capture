# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 16:00:40 2020

@author: ning
"""
import os
import PIL

import numpy as np

from matplotlib import pyplot as plt
from glob       import glob

output_dir = '../results'
figure_dir = '../figures'
if not os.path.exists(figure_dir):
    os.mkdir(figure_dir)

directories             = {}
for name in ['body','face','hand']:
    temp                = os.path.join(output_dir,name)
    directories[name]   = temp.replace('\\','/')

array_files             = {}
for name in ['body','face','hand']:
    temp                = np.sort(glob(os.path.join(directories[name],
                                                    '*',
                                                    '*.npy')))
    array_files[name]   = temp

collections             = np.array(list(array_files.values()))

for (body_array,face_array,hand_array) in collections.T:
    body                = np.load(body_array)
    face                = np.load(face_array)
    hands               = np.load(hand_array)
    
    body_array          = body_array.replace('\\','/')
    video_name          = body_array.split('/')[-2]
    image_name          = body_array.split('/')[-1].replace('.npy','.jpeg').replace('frame_','')
    
    imagePath           = os.path.join(output_dir,'frames',video_name,image_name)
    image_array         = np.array(PIL.Image.open(imagePath))
    figure_saving_dir   = os.path.join(figure_dir,video_name)
    width               = int(image_array.shape[1] / 100)
    height              = int(image_array.shape[0] / 100)
    if not os.path.exists(figure_saving_dir):
        os.mkdir(figure_saving_dir)
    
    plt.close("all")
    fig,ax              = plt.subplots(figsize = (width,height))
    ax.imshow(image_array)
    for ii,person in enumerate(body): # in case more than one person was detected
        person      = person[person[:,-1] > 0]
        person_face = face[ii,:,:]
        
        ax.scatter(person[:,0],
                   person[:,1],
                   10,
                   color = 'blue',
                   )
        ax.scatter(person_face[:,0],
                   person_face[:,1],
                   10,
                   color='red',
                   )
        for hand in hands: # in case only one hand was detected
            ax.scatter(hand[0,:,0],
                       hand[0,:,1],
                       10,
                       color='yellow',
                       )
    fig.savefig(os.path.join(figure_saving_dir,image_name))
    plt.close("all")
