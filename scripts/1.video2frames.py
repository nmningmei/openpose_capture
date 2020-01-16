# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 11:31:17 2020

@author: ning
"""

import os
import cv2
from glob import glob

working_dir = '../data'
working_data = glob(os.path.join(working_dir,'*.mp4'))
saving_dir = '../results/frames'
if not os.path.exists(saving_dir):
    os.makedirs(saving_dir)

for video_name in working_data:
    video_name = video_name.replace('\\','/')
    add_to_saving_name = video_name.split('/')[-1].split('.')[0]
    if not os.path.exists(os.path.join(saving_dir,add_to_saving_name)):
        os.mkdir(os.path.join(saving_dir,add_to_saving_name))
    # Start capturing the feed
    cap = cv2.VideoCapture(video_name)
    # Find the number of frames
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    print ("Number of frames: ", video_length)
    count = 0
    print ("Converting video..\n")
    # Start converting the video
    while cap.isOpened():
        # Extract the frame
        ret, frame = cap.read()
        # Write the results back to output location.
        cv2.imwrite(os.path.join(saving_dir,
                                 add_to_saving_name,
                                 f"{count + 1}.jpeg"), 
                    frame)
        count = count + 1
        # If there are no more frames left
        if (count > (video_length-1)):
            # Release the feed
            cap.release()
            # Print stats
            print (f"Done extracting frames.\n{count} frames extracted")
            break