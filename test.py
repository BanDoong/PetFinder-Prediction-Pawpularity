import os
import pandas as pd
from torchvision import models
import torch
import torch.nn as nn

# from model import EfficientHybridviT

from torchinfo import summary

# model = EfficientHybridviT()
from efficientnet_pytorch import EfficientNet

model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=1)

from model import EfficientHybridviT
from model import EfficientHybridSwin


summary(EfficientHybridSwin(base=model), (4, 3, 896, 896))
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, Sequential, Input, layers, applications

"""
BASE_WEIGHTS = './tf_weights/'

IMG_NET = [
    'imagenet/imagenet.notop-b0.h5',
    'imagenet/imagenet.notop-b1.h5',
    'imagenet/imagenet.notop-b2.h5'
]

NOISY_STUD = [
    'noisystudent/noisy.student.notop-b0.h5',
    'noisystudent/noisy.student.notop-b1.h5',
    'noisystudent/noisy.student.notop-b2.h5',
]
print(BASE_WEIGHTS + NOISY_STUD[0])
# inputx = np.randn((896, 896, 3))
img_size = 896
inputx = Input((img_size, img_size, 3), name='input_hybrids')
base = applications.EfficientNetB0(
    include_top=False,
    weights=BASE_WEIGHTS + NOISY_STUD[0],
    input_tensor=inputx
)
# base model with compatible output which will be an input of transformer model
new_base = Model(
    [base.inputs],
    [
        base.get_layer('block1a_activation').output,
        base.output
    ],  # output with 192 feat_maps
    name='efficientnet'
)
new_base.summary()
"""
