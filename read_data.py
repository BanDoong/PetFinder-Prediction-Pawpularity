import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import pandas as pd
from main import fix_seed

fix_seed()


def using_meta(meta, index):
    focus = meta['Subject Focus'][index]
    eye = meta['Eyes'][index]
    face = meta['Face'][index]
    near = meta['Near'][index]
    action = meta['Action'][index]
    accessory = meta['Accessory'][index]
    group = meta['Group'][index]
    collage = meta['Collage'][index]
    human = meta['Human'][index]
    occ = meta['Occlusion'][index]
    info = meta['Info'][index]
    blur = meta['Blur'][index]

    out = [focus, eye, face, near, action, accessory, group, collage, human, occ, info, blur]

    return out


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transforms, dataset='training', model=None):
        self.dataset = dataset
        self.model = model
        self.meta = pd.read_csv('./petfinder-pawpularity-score/train.csv')
        if self.dataset == 'training':
            self.df = pd.read_csv('./petfinder-pawpularity-score/train_label.csv')
        else:
            self.df = pd.read_csv('./petfinder-pawpularity-score/val_label.csv')

        self.df['Pawpularity'] /= 100.0
        self.data_dir = data_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        if self.model == 'hybrid' or self.model == 'hybrid_2' or self.model == 'hybrid_swin':
            img_path = os.path.join(self.data_dir, 'train_resized_896', self.df['Id'][index]) + '.jpg'
        else:
            img_path = os.path.join(self.data_dir, 'train_resized', self.df['Id'][index]) + '.jpg'
        meta_data = np.asarray(using_meta(self.meta, index))
        img = Image.open(img_path)
        img = self.transforms(img)
        label = torch.tensor(self.df['Pawpularity'][index], dtype=torch.float32).reshape(1)

        sample = {'img': img, 'label': label, 'meta': torch.from_numpy(meta_data)}
        return sample


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transforms):
        self.df = pd.read_csv('./petfinder-pawpularity-score/test.csv')
        self.data_dir = data_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = os.path.join(self.data_dir, 'test', self.df['Id'][index]) + '.jpg'
        img = Image.open(img_path)
        img = self.transforms(img)
        sample = {'img': img, 'Id': self.df['Id']}
        return sample


class Normalization(object):
    def __init__(self, normal=True):
        self.normal = normal

    def __call__(self, img):
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        return img


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, img):
        img = img.transpose((2, 0, 1)).astype(np.float32)
        return img


class RandomFlip(object):
    def __call__(self, img):

        if np.random.rand() > 0.5:
            img = np.fliplr(img)
        if np.random.rand() > 0.5:
            img = np.fliplr(img)

        return img
