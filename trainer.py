import wandb
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score, mean_absolute_error

from tqdm import tqdm
from copy import deepcopy

class Trainer():
    def __init__(self, train_args, train_loader, valid_loader, model, optimizer, criterion, lr_scheduler, device, logger):
        self.train_args = train_args
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.model = model
        self.optimizer = optimizer # AdamW
        self.criterion = criterion # nn.MSELoss()
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.logger = logger

        wandb.init(project='Muscles in action(Train)')
        wandb.run.name = train_args.name
        wandb.run.save()

        super().__init__()

    def get_learning_rate(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def load_data(self, data_retval):
        skeleton = data_retval['3dskeleton']

        if skeleton.shape[-1] == 2:
            # 2차원 스켈레톤은 좌표값이 0~1로 정규화가 안되어있어서 정규화하는 코드
            divide = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.tensor([1080.0,1920.0]),dim=0),dim=0),dim=0).repeat(skeleton.shape[0],skeleton.shape[1],skeleton.shape[2],1).to(self.device)
            skeleton = skeleton / divide

        skeleton = skeleton.reshape(skeleton.shape[0], skeleton.shape[1], -1)
        skeleton = skeleton.to(self.device)
        
        emg_gt = data_retval['emg_values'].to(torch.float32).to(self.device)
        cond = data_retval['condval'].to(torch.float32).to(self.device)
                
        return skeleton, emg_gt, cond

    def train_one_epoch(self, epoch):
        self.model.train()
        
        log_str = f'Epoch (1-based): {epoch + 1} / {self.train_args.num_epochs}'
        self.logger.info()
        self.logger.info('=' * len(log_str))
        self.logger.info(log_str)
        self.logger.info(f'===> Train')
        # self.logger.report_scalar('train/learn_rate', self.get_learning_rate(self.optimizer), step=epoch)

        total_loss = 0
        for data_retval in tqdm(self.train_loader):
            skeleton, emg_gt, cond = self.load_data(data_retval)

            emg_pred = self.model(skeleton, cond)

            loss = self.criterion(emg_pred, emg_gt)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

            total_loss += float(loss)

        return total_loss / len(self.train_loader)
    
    def validate(self):
        self.model.eval()

        self.logger.info(f'===> Validation')
        with torch.no_grad():
            total_loss = 0
            total_mae = 0
            total_r2 = 0
            for data_retval in tqdm(self.valid_loader):
                skeleton, emg_gt, cond = self.load_data(data_retval)
                
                emg_pred = self.model(skeleton, cond)

                loss = self.criterion(emg_pred, emg_gt)
                mae = mean_absolute_error(emg_gt.cpu().flatten(), emg_pred.cpu().flatten())
                r2 = r2_score(emg_gt.cpu().flatten(), emg_pred.cpu().flatten())

                total_loss += float(loss)
                total_mae += float(mae)
                total_r2 += float(r2)

            wandb.log({'Validation MAE': total_mae / len(self.valid_loader)})
            wandb.log({'Validation R2': total_r2 / len(self.valid_loader)})
            
            return total_loss / len(self.valid_loader)

    def train(self, start_epoch):
        lowest_loss = np.inf
        best_model = None
        patience = 10

        self.logger.info('Start training loop...')
        for epoch in range(start_epoch, self.train_args.num_epochs):
            train_loss = self.train_one_epoch(epoch)
            wandb.log({'Training loss': train_loss})
            valid_loss = self.validate()
            wandb.log({'Validation loss': valid_loss})

            if valid_loss <= lowest_loss:
                lowest_loss = valid_loss
                best_model = deepcopy(self.model.state_dict())
                patience = 10
            
            else:
                patience -= 1
            
            self.logger.info(f'Epoch {epoch + 1}: train_loss={train_loss}  valid_loss={valid_loss}  lowest_loss={lowest_loss}')

            if patience == 0:
                self.logger.info(f'Early stopping in epoch {epoch + 1}')
                break
        
        self.model.load_state_dict(best_model)

