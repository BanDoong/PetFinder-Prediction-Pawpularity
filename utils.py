from __future__ import absolute_import, division, print_function

import os
import logging
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F


def RMSELoss(yhat, y):
    return torch.sqrt(torch.mean((yhat - y) ** 2))


class Parser:
    def __init__(self, parser):
        self.__parser = parser
        self.__args = parser.parse_args()

    def get_parser(self):
        return self.__parser

    def get_arguments(self):
        return self.__args


class Logger:
    def __init__(self, info=logging.INFO, name=__name__):
        logger = logging.getLogger(name)
        logger.setLevel(info)

        self.__logger = logger

    def get_logger(self, handler_type='stream_handler'):
        if handler_type == 'stream_handler':
            handler = logging.StreamHandler()
            log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(log_format)
        else:
            handler = logging.FileHandler('utils.log')

        self.__logger.addHandler(handler)

        return self.__logger


class EarlyStopping:
    """주어진 patience 이후로 validation loss가 개선되지 않으면 학습을 조기 중지"""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): validation loss가 개선된 후 기다리는 기간
                            Default: 7
            verbose (bool): True일 경우 각 validation loss의 개선 사항 메세지 출력
                            Default: False
            delta (float): 개선되었다고 인정되는 monitered quantity의 최소 변화
                            Default: 0
            path (str): checkpoint저장 경로
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''validation loss가 감소하면 모델을 저장한다.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
