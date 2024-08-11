import time
import os
import random
import torch
import torchvision
import numpy as np

from logger import CustomLogger
from data import create_dataloaders
from args import train_args
from model import TransformerEnc
from trainer import Trainer

def main(args, logger):
    logger.info()
    logger.info('torch version:', str(torch.__version__))
    logger.info('torchvision version:', str(torchvision.__version__))
    logger.save_args(args)

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    if args.device == 'mps':
        torch.mps.manual_seed(args.seed)
    
    device = torch.device(args.device)
    args.checkpoint_path = args.checkpoint_path + '/' + args.name

    logger.info('Checkpoint path:', args.checkpoint_path)
    os.makedirs(args.checkpoint_path, exist_ok=True)

    logger.info('Initializing data loaders...')
    start_time = time.time()
    train_loader, valid_loader, dataset_args = create_dataloaders(args, logger)
    logger.info(f'Took {time.time() - start_time:.3f}s')

    logger.info('Initializing model...')
    start_time = time.time()

    model_args = {
                  'dim_model': int(args.dim_model),
                  'num_heads': int(args.num_heads),
                  'num_encoder_layers':int(args.num_encoder_layers),
                  'dropout_p':float(args.dropout_p),
                  'device': args.device
                  }
    
    model = TransformerEnc(**model_args)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learn_rate)
    criterion = torch.nn.MSELoss()
    milestones = [(args.num_epochs * 2) // 5,
                  (args.num_epochs * 3) // 5,
                  (args.num_epochs * 4) // 5]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=args.lr_decay)

    if args.resume:
        logger.info('Loading weights from:', args.resume)
        checkpoint = torch.load(args.resume, map_location=args.device)
        model_dict = checkpoint['model']
        for key in list(model_dict.keys()):
            model_dict[key[7:]] = model_dict.pop(key)
        model.load_state_dict(model_dict)

    trainer = Trainer(args, train_loader, valid_loader, model, optimizer, criterion, lr_scheduler, device, logger)
    
    logger.info(f'Took {time.time() - start_time:.3f}s')

    trainer.train(start_epoch=0)
    
    logger.info(f'Saving model checkpoint to {args.checkpoint_path}...')
    torch.save({
        'model': trainer.model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
    }, os.path.join(args.checkpoint_path, 'best_model.pth'))


if __name__ == '__main__':
    np.set_printoptions(precision=3, suppress=True)
    torch.set_printoptions(precision=3, sci_mode=False)

    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.cuda.empty_cache()

    args = train_args()

    logger = CustomLogger(args, args, context='train')

    try:
        main(args, logger)
    
    except Exception as e:
        logger.exception(e)
        logger.warning('Shutting down due to exception...')