import numpy as np
import random
import torch
import os

torch.backends.cudnn.deterministic = True
random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)



class MuscleDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, logger, phase, percent, step, cond, weight=None, transform=None):
        phase_dir = os.path.join(dataset_path, phase)
        if not os.path.exists(phase_dir):
            phase_dir = dataset_path
        
        self.phase = phase
        with open(dataset_path) as f:
            if not weight:
                lines = f.readlines()
            else:
                lines = [line for line in f.readlines() if weight in line]
        
        self.files = lines
        self.file_count = len(self.files)
        print('File count:', self.file_count)

        self.dataset_path = dataset_path
        self.logger = logger
        self.phase = phase
        self.phase_dir = phase_dir
        self.cond = cond
        self.transform = transform # ?
        self.percent = float(percent) # ?
        self.step = int(step)
        self.maxemg = 100
        self.bins = np.linspace(0, self.maxemg, 20)
        self.log_dir = 'training_viz_digitized'
        self.plot = False
        self.muscles = ['ErectorSpinae', 'UpperTrapezius', 'BicepsBrachii', 'FlexorDigitorumSuperficialis', 'ExtensorDigitorum', 'BicepsFemoris', 'VastusLateralis', 'ExternalOblique']
    
    
    def __len__(self):
        return int(self.file_count * self.percent)
    

    def __getitem__(self, idx):
        filepath = self.files[idx].split("\n")[0]
        # emg value
        emg_values = np.round(np.load(filepath + "/emg_values.npy"))
        # 3d joints
        threed_joints = np.load(filepath + "/3d_skeleton.npy")

        person = filepath.split("/")[2]
        
        subject_dict = {
            'Subject0': {'condval': np.array([1.0])},

            'Subject1': {'condval': np.array([0.85])},

            'Subject2': {'condval': np.array([0.68])},

            'Subject3': {'condval': np.array([0.51])},

            'Subject4': {'condval': np.array([0.34])},

            'Subject5': {'condval': np.array([0.17])}
        }
        
        if self.cond:
            condval = subject_dict[person]['condval']
        
        result = {'condval':condval,
                'emg_values': emg_values.transpose(1,0),
                'filepath': filepath,
                '3dskeleton': threed_joints[:,:,:],
                'indexpath': filepath.split("/")[-1]}
        
        return result

def create_dataloaders(args, logger):
    '''
    return (train_loader, val_aug_loader, val_no_aug_loader, dataset_args)
    '''
    dataset_args = {'percent': args.percent,
                    'step': int(args.step),
                    'cond': args.cond}
    
    train_dataset = MuscleDataset(dataset_path=args.data_path_train, weight=args.weight, logger=logger,
                                  phase='train', **dataset_args)
    valid_dataset = MuscleDataset(dataset_path=args.data_path_val, weight=args.weight, logger=logger,
                                  phase='val', **dataset_args)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs, num_workers=args.num_workers,
                                               shuffle=True, worker_init_fn=seed_worker, drop_last=True, pin_memory=False)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.bs, num_workers=args.num_workers,
                                               shuffle=False, worker_init_fn=seed_worker, drop_last=True, pin_memory=False)
    
    return train_loader, valid_loader, dataset_args
