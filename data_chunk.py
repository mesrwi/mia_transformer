import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

os.chdir('dataset/resamoe=')

tasks = ['5kg_ground', '5kg_move', '5kg_top', '10kg_ground', '10kg_move', '10kg_top']
for i in range(6):
    subject = f'Subject{i}'
    for task in tasks:
        frames = sorted([int(f) for f in os.listdir(os.path.join(os.getcwd(), subject, task)) if not f.startswith('.')])

        chunk_id = 0
        for frame in frames:
            interval = [frame, frame + 1, frame + 2]
            if (interval[1] in frames) and (interval[2] in frames):
                filepath = f'/Users/mesrwi/kitech/project_skl/mia_transformer/dataset/chunk/{subject}/{task}/{chunk_id}'
                if not os.path.exists(filepath):
                    os.makedirs(filepath)

                skl1 = np.load(f'{os.path.join(os.getcwd(), subject, task, str(interval[0]))}/3d_skeleton.npy')
                skl2 = np.load(f'{os.path.join(os.getcwd(), subject, task, str(interval[1]))}/3d_skeleton.npy')
                skl3 = np.load(f'{os.path.join(os.getcwd(), subject, task, str(interval[2]))}/3d_skeleton.npy')
                skl = np.concatenate([skl1, skl2, skl3], axis=0)

                np.save(f'{filepath}/3d_skeleton.npy', skl)

                emg1 = np.load(f'{os.path.join(os.getcwd(), subject, task, str(interval[0]))}/emg_value.npy')
                emg2 = np.load(f'{os.path.join(os.getcwd(), subject, task, str(interval[1]))}/emg_value.npy')
                emg3 = np.load(f'{os.path.join(os.getcwd(), subject, task, str(interval[2]))}/emg_value.npy')
                emg = np.concatenate([emg1, emg2, emg3], axis=0)

                np.save(f'{filepath}/emg_value.npy', emg)

                chunk_id += 1