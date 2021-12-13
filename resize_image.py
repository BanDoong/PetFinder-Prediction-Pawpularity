# from tensorflow.keras.layers import Conv3D, Conv3DTranspose, Dense, Flatten, Dropout, BatchNormalization, Reshape, \
#     LeakyReLU, Input
# from tensorflow.keras import Model, Sequential

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import os
from tqdm import tqdm
from matplotlib import pyplot as plt
from PIL import Image


def getListOfFiles(dirName):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles


# train_files = getListOfFiles('petfinder-pawpularity-score/train')
# print(train_files)
# train_df = pd.read_csv('petfinder-pawpularity-score/train.csv').set_index('Id')
# train_ids = [os.path.basename(file).split('.')[0] for file in train_files]
#
# test_files = getListOfFiles('petfinder-pawpularity-score/test')
# y_train = train_df.loc[train_ids].Pawpularity
path = './petfinder-pawpularity-score/train'
train_files = os.listdir(path)
train_files.sort()

def read_resize(file, size):
    file = os.path.join(path, file)
    x = tf.io.read_file(file)
    x = tf.io.decode_jpeg(x)
    x = tf.keras.preprocessing.image.smart_resize(x, size, interpolation='bilinear')
    x = x.numpy().astype(np.uint8)
    return x


# x_train_256 = np.stack([read_resize(file, (256, 256)) for file in tqdm(train_files)])
# x_train_256.nbytes / 1024 / 1024 / 1024

for file in tqdm(train_files):
    im = Image.fromarray(read_resize(file, (896, 896))).convert('RGB')
    im.save(f'petfinder-pawpularity-score/train_resized_896/{file}')
