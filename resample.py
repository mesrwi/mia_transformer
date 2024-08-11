import os
import json
import pandas as pd
import numpy as np

def emg_smoothing(emg_signal, target_rate):
    # 시작 시간을 0으로 초기화
    emg_signal.iloc[:, 0] -= emg_file.iloc[0, 0]

    # 타임스탬프 형식으로 변환 후, 인덱스로 설정
    emg_signal.iloc[:, 0] = pd.to_timedelta(emg_signal.iloc[:, 0], unit='s')
    emg_signal.set_index(emg_signal.columns[0], inplace=True)

    # 리샘플링된 데이터의 시간 간격을 fps를 기준으로 계산
    factor = int(100 / target_rate * 10)
    resampled_emg = emg_file.resample(f'{factor}L').mean()

    return resampled_emg


def skeleton_resampling(skeleton, original_fps, target_fps):
    # 데이터 포인트의 수 설정
    num_points = len(skeleton)
    
    # 각 데이터 포인트에 대한 시간 간격 계산
    time_interval = 1 / original_fps

    # 시간 배열 생성
    time_array = np.arange(0, num_points * time_interval, time_interval)

    # 시간 배열을 Timedelta로 변환
    timedeltas = pd.to_timedelta(time_array, unit='s')

    # 데이터프레임 생성
    skl_df = pd.DataFrame(skeleton.reshape(-1, 17*3))
    try:
        skl_df['Time'] = timedeltas
    except:
        skl_df['Time'] = timedeltas[:len(skl_df)]

    # 타임스탬프를 인덱스로 설정
    skl_df.set_index('Time', inplace=True)

    # 리샘플링된 데이터의 시간 간격을 fps를 기준으로 계산
    factor = int(100 / target_fps * 10)
    resampled_skl = skl_df.resample(f'{factor}L').mean().values.reshape(-1, 17, 3)

    return resampled_skl


def save_in_chunks(data, chunk_size, stride, save_dir, filename):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    num_data = data.shape[0]
    for i in range(0, num_data, stride):
        start_index = i
        end_index = start_index + chunk_size
        chunk_data = data[start_index: end_index]

        if len(chunk_data) < chunk_size:
            break

        chunk_dir = os.path.join(save_dir, f'{i // 5 + 1}')
        if not os.path.exists(chunk_dir):
            os.makedirs(chunk_dir)
        
        filepath = os.path.join(chunk_dir, f'{filename}')

        np.save(filepath, chunk_data)


with open('sync_data.json', 'r') as f:
    meta = json.load(f)

raw_path = 'dataset/raw'
out_dir = 'dataset/resampled'

for i in range(6):
    subject = f'Subject{i}'
    sub_path = os.path.join(raw_path, subject)

    emg_files = sorted(os.listdir(os.path.join(sub_path, 'emg')))
    skl_files = sorted(os.listdir(os.path.join(sub_path, '3d_skeleton')))
    
    for emg_filename, skl_filename in zip(emg_files, skl_files):
        print(subject, emg_filename, skl_filename)
        emg_file = pd.read_csv(f'{sub_path}/emg/{emg_filename}', index_col=0)
        smoothed_emg = emg_smoothing(emg_file, 25)

        skl = np.load(f'{sub_path}/3d_skeleton/{skl_filename}')
        fps = meta[subject][skl_filename[:-4]]['fps']
        
        resampled_skl = skeleton_resampling(skl, fps, 25)

        # 데이터 개수 매치
        if len(smoothed_emg) < resampled_skl.shape[0]:
            resampled_skl = resampled_skl[:len(smoothed_emg), :, :]
        elif len(smoothed_emg) > resampled_skl.shape[0]:
            smoothed_emg = smoothed_emg.iloc[:resampled_skl.shape[0], :]

        save_in_chunks(smoothed_emg.values, chunk_size=125, stride=5, save_dir=os.path.join(out_dir, subject, skl_filename[:-4]), filename='emg_values.npy')
        save_in_chunks(resampled_skl, chunk_size=125, stride=5, save_dir=os.path.join(out_dir, subject, skl_filename[:-4]), filename='3d_skeleton.npy')

        