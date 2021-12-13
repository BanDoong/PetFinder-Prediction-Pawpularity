import argparse
from train import *
from read_data import *
from utils import *
import random
import torch.backends.cudnn as cudnn

"""
python  main.py \
        --mode train \
        --modality mri tau \
        --dir_label getlabels_AD_CN \
        --num_label 2 \ 
        --num_workers 2 \
        --batch_size 4
        --epochs 100 \
        --lr 1e-4 \
        --group classification \
        --dir_data ./caps/subjects \
        --dir_ckpt ./ckpt \
"""


## Reproducibility

# # torch random seed 고정 ==> reproducible
# torch.manual_seed(1)
# torch.cuda.manual_seed(2)
# torch.cuda.manual_seed_all(3) # if use multi-GPU
# # cudnn randomness 고정
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# # numpy random 고정
# np.random.seed(4)
# # torchvision random 고정
# random.seed(5)
def fix_seed():
    # torch random seed 고정 ==> reproducible
    torch.manual_seed(1)
    torch.cuda.manual_seed(2)
    torch.cuda.manual_seed_all(3)  # if use multi-GPU
    # cudnn randomness 고정
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    # numpy random 고정
    np.random.seed(4)
    # torchvision random 고정
    random.seed(50)


fix_seed()

parser = argparse.ArgumentParser(description='Train the my Network',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--model', default='vit_b_16',
                    choices=['vit_b_16', 'vit_b_32', 'vit_L_16', 'resnet', 'swin', 'effinet', 'effinet_b5', 'use_meta',
                             'ensemble', 'vit_scheduler', 'hybrid', 'hybrid_2', 'hybrid_swin'], dest='model')
parser.add_argument('--num_epoch', type=int, default=100, dest='num_epoch')
parser.add_argument('--batch_size', type=int, default=128, dest='batch_size')
parser.add_argument('--lr', default=1e-4, dest='lr', type=float, help='learning rate')
parser.add_argument('--num_worker', type=int, default=12, dest='num_worker')
parser.add_argument('--weight_decay', default=1e-5, dest='weight_decay')

parser.add_argument('--dir_data', default='./petfinder-pawpularity-score', dest='dir_data')
parser.add_argument('--dir_ckpt', default='./ckpt', dest='dir_ckpt')
parser.add_argument('--dir_log', default='./logs', dest='dir_log')
parser.add_argument('--gpus', default=0, type=int, dest='gpus')
parser.add_argument('--patience', default=10, type=int, dest='patience')
parser.add_argument('--use_meta', action='store_true', dest='use_meta')
# output directory
parser.add_argument('--dir_out', default='./output', dest='dir_out')

PARSER = Parser(parser)


def main():
    ARGS = PARSER.get_arguments()

    TRAINER = Train(ARGS)

    TRAINER.train()


if __name__ == '__main__':
    main()
