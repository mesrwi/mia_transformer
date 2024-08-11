import yaml
import argparse

def train_args():
    with open('/Users/mesrwi/kitech/project_skl/mia_transformer/configs/train.yaml', 'r') as f:
        args_dict = yaml.load(f, Loader=yaml.FullLoader)
    parser = argparse.ArgumentParser()

    for key in args_dict.keys():
        parser.add_argument(f"--{key}", default=args_dict[key])
    
    args = parser.parse_args()
    args.bs = int(args.batch_size)
    args.learn_rate = float(args.learning_rate)
    args.modelname = args.modelname.split('_')[0]

    return args

def inference_args():
    pass

def verify_args(args, is_train=False):
    pass