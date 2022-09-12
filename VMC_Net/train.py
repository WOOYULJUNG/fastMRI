import torch

#CUDA_LAUNCH_BLOCKING = 1
import argparse
import shutil
import os, sys
if os.getcwd() + '/utils/model/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/model/')

from utils.learning.train_part import train_all
from pathlib import Path


def parse():
    parser = argparse.ArgumentParser(description='Train Unet on FastMRI challenge Images',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--GPU-NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('-e', '--num-epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('-r', '--report-interval', type=int, default=500, help='Report interval')
    parser.add_argument('-n', '--net-name', type=Path, default='VMC_Net', help='Name of network')
#    parser.add_argument('-t', '--data-path-train', type=Path, default='/root/input/train', help='Directory of train data')
#    parser.add_argument('-v', '--data-path-val', type=Path, default='/root/input/val', help='Directory of validation data')
#    parser.add_argument('-t', '--data-path-train', type=Path, default='/content/content/gdrive/Shareddrives/Ummmies/dataset/train', help='Directory of train data')
#    parser.add_argument('-v', '--data-path-val', type=Path, default='/content/content/gdrive/Shareddrives/Ummmies/dataset/val', help='Directory of validation data')
    parser.add_argument('-t', '--data-path-train', type=Path, default='/content/dataset/train', help='Directory of train data')
    parser.add_argument('-v', '--data-path-val', type=Path, default='/content/dataset/val', help='Directory of validation data')

    parser.add_argument('--input-key', type=str, default='kspace', help='Name of input key')
    parser.add_argument('--target-key', type=str, default='image_label', help='Name of target key')
    parser.add_argument('--max-key', type=str, default='max', help='Name of max key in attributes')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse()
    args.exp_dir_varnet = '../result' / args.net_name / 'varnet' / 'checkpoints'
    args.exp_dir_multidomain = '../result' / args.net_name / 'multidomain' / 'checkpoints'
    args.exp_dir_crossdomain7 = '../result' / args.net_name / 'crossdomain7' / 'checkpoints'
    args.exp_dir_crossdomain8 = '../result' / args.net_name / 'crossdomain8' / 'checkpoints'
    args.exp_dir_ensemble = '../result' / args.net_name / 'ensemble' / 'checkpoints'
    args.val_dir = '../result' / args.net_name / 'reconstructions_val'
    args.main_dir = '../result' / args.net_name / __file__
    
    args.exp_dir_varnet.mkdir(parents=True, exist_ok=True)
    args.exp_dir_multidomain.mkdir(parents=True, exist_ok=True)
    args.exp_dir_crossdomain7.mkdir(parents=True, exist_ok=True)
    args.exp_dir_crossdomain8.mkdir(parents=True, exist_ok=True)
    args.exp_dir_ensemble.mkdir(parents=True, exist_ok=True)
    args.val_dir.mkdir(parents=True, exist_ok=True)

    train_all(args)
