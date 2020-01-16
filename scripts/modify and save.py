# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 14:56:23 2020

@author: ning
"""

import os
import re
import numpy as np
from glob import glob
frames_dir = "../results/frames"
allImagePaths = np.sort(glob(os.path.join(frames_dir,'*','*.jpeg')))

templates = ['2.capture body.py','3.capture face.py','4.capture hand.py']

bash_folder = 'body_face_hand_bash'
if not os.path.exists(bash_folder):
    os.mkdir(bash_folder)

collection = []
for template in templates:
    for ii,single_image in enumerate(allImagePaths):
        single_image = single_image.replace('\\','/')
        frame_index = re.findall('\d+',single_image)[-1]
        video_folder_name = single_image.split('/')[-2]
        new_script_name = os.path.join(bash_folder,template.replace('.py',
                                f'_{video_folder_name}_frame_{frame_index}.py'))
        new_script_name = new_script_name.replace('\\','/')
        with open(new_script_name,'w') as new_file:
            with open(template,'r') as old_file:
                for line in old_file:
                    if "# batch change" in line:
                        line = line.replace('0',f'{ii}')
                    elif '../' in line:
                        line = line.replace('../','../../')
                    new_file.write(line)
                old_file.close()
            new_file.close()
        collection.append(new_script_name)
collection = np.array(collection).reshape(len(templates),-1).T

with open(f'{bash_folder}/run_all.py','w') as f:
    f.write("""
import os
import time""")
    f.close()
with open(f'{bash_folder}/run_all.py','a') as f:
    for line in collection:
        print()
        for component in ['capture body','capture face','capture hand']:
            script_picked = [item for item in line if (component in item)][0]
            print(script_picked)
            python_command = f"python '{script_picked.split('/')[-1]}'"
            f.write(f'\nos.system("{python_command}")\ntime.sleep(3)\n')













