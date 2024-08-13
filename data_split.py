import os
import shutil
from random import sample

os.chdir('dataset/resampled_stride25')

tasks = ['5kg_ground', '5kg_move', '5kg_top', '10kg_ground', '10kg_move', '10kg_top']
for i in range(6):
    subject = f'Subject{i}'
    for task in tasks:
        files = sorted([int(file) for file in os.listdir(f'{subject}/{task}') if not file.startswith('.')])
        # val_files = sample(files, int(len(files) * 0.2))
        # train_files = [file for file in files if file not in val_files]
        val_files = files[int(len(files)*0.8):]
        train_files = files[:int(len(files)*0.8)]

        for file in val_files:
            from_file_path = f'{subject}/{task}/{file}'
            to_file_path = f'dataset/val/{subject}/{task}/{file}'

            shutil.copytree(from_file_path, to_file_path)

        for file in train_files:
            from_file_path = f'{subject}/{task}/{file}'
            to_file_path = f'dataset/train/{subject}/{task}/{file}'

            shutil.copytree(from_file_path, to_file_path)

# 불러와야 할 파일 경로를 txt 파일로 지정
train_path = 'dataset/train'
val_path = 'dataset/val'
    
for i in range(6):
    subject = f'Subject{i}'
    for task in tasks:
        with open('dataset/datasetsplits/train.txt', 'a') as f:
            for id in sorted([int(ids) for ids in os.listdir(os.path.join(train_path, subject, task))]):
                f.write(os.path.join('dataset/train', subject, task, str(id))+'\n')

        with open('dataset/datasetsplits/val.txt', 'a') as f:
            for id in sorted([int(ids) for ids in os.listdir(os.path.join(val_path, subject, task))]):
                f.write(os.path.join('dataset/val', subject, task, str(id))+'\n')