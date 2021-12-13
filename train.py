# coding: utf8

from torch import nn
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from read_data import *
from tqdm import tqdm
from time import time
from torchvision import transforms
from torchvision.transforms import Resize
# from model import *
from easydict import EasyDict as edict
import pandas as pd
from pytorch_pretrained_vit import ViT
from utils import RMSELoss
import ssl
from torchvision import models
from sklearn.linear_model import LinearRegression
from utils import EarlyStopping

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context

# from apex.parallel import DistributedDataParallel as DDP
# MULTI GPU
from main import fix_seed

fix_seed()


class Train:
    def __init__(self, args):
        self.model = args.model
        self.num_epoch = args.num_epoch
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.num_worker = args.num_worker
        self.weight_decay = args.weight_decay
        self.dir_data = args.dir_data
        self.dir_ckpt = args.dir_ckpt
        self.dir_log = args.dir_log
        self.gpus = args.gpus
        self.dir_out = args.dir_out
        self.use_meta = args.use_meta
        self.patience = args.patience

        self.device = torch.device(f'cuda:{self.gpus}' if torch.cuda.is_available() else 'cpu')

    def train(self):
        lr = self.lr
        weight_decay = self.weight_decay
        batch_size = self.batch_size
        device = self.device
        num_epoch = self.num_epoch
        num_worker = self.num_worker
        dir_data = self.dir_data
        dir_out = self.dir_out
        fn_loss_MSE = nn.MSELoss().to(device)

        def to_cpu(data):
            return data.cpu().clone().detach().numpy()

        writer_train = SummaryWriter(f'{self.dir_ckpt}/{self.model}')

        def save(path, net=None, optim=None, pretrain=None):
            if pretrain:
                torch.save({'net': net.state_dict(), 'optim': optim.state_dict()}, path)
            else:
                torch.save({'net': net.state_dict(), 'optim': optim.state_dict()}, path)

        ## 네트워크 생성
        # CPU GPU 결정 위해 to Device 사용

        if self.model == 'vit_b_16':
            net = ViT('B_16', pretrained=True, num_classes=1)
        elif self.model == 'vit_b_32':
            net = ViT('B_32', pretrained=True, num_classes=1)
        elif self.model == 'vit_L_16':
            net = ViT('L_16', pretrained=True, num_classes=1)
        elif self.model == 'resnet':
            net = models.resnext50_32x4d(pretrained=True, progress=True)
            num_ftrs = net.fc.in_features
            net.fc = nn.Linear(num_ftrs, 1)
        elif self.model == 'effinet':
            from efficientnet_pytorch import EfficientNet
            net = EfficientNet.from_pretrained('efficientnet-b0', num_classes=1)
            # num_ftrs = net.classifier[1].in_features
            # net.fc = nn.Linear(num_ftrs, 1)
        elif self.model == 'effinet_b5':
            from efficientnet_pytorch import EfficientNet
            net = EfficientNet.from_pretrained('efficientnet-b5', num_classes=1)

        elif self.model == 'ensemble':
            net = models.efficientnet_b0(pretrained=True, progress=True)
            # num_ftrs = net.classifier[1].in_features
            net.classifier[1].out_features = 1
            net_vit = ViT('B_16', pretrained=True, num_classes=1)
            net_vit.to(device)
        elif self.model == 'use_meta':
            from efficientnet_pytorch import EfficientNet
            net = EfficientNet.from_pretrained('efficientnet-b7', num_classes=12)
            net_vit = ViT('B_16', pretrained=True, num_classes=12)
            net_vit.to(device)
            # net = ViT('B_16', pretrained=True, num_classes=12)
            from model import meta_model
            model = meta_model()
            model.to(device)

        elif self.model == 'hybrid':
            from model import EfficientHybridviT
            from efficientnet_pytorch import EfficientNet
            model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=1)
            net = EfficientHybridviT(base=model)

        elif self.model == 'hybrid_2':
            from model import EfficientHybridviT_2
            from efficientnet_pytorch import EfficientNet
            model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=1)
            net = EfficientHybridviT_2(base=model)

        elif self.model == 'hybrid_swin':
            from model import EfficientHybridSwin
            from efficientnet_pytorch import EfficientNet
            model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=1)
            net = EfficientHybridSwin(base=model)

        elif self.model == 'vit_scheduler':
            net = ViT('B_16', pretrained=True, num_classes=1)

        if (device.type == 'cuda') and (torch.cuda.device_count() > 1):
            print("Multi GPU ACTIVATES")
            if self.model == 'use_meta':
                net = nn.DataParallel(net)
                net_vit = nn.DataParallel(net_vit)
                model = nn.DataParallel(model)
            elif self.model == 'hybrid' or self.model == 'hybrid_2' or self.model == 'hybrid_swin':
                net = nn.DataParallel(net)
        net.to(device)

        params_to_mri = [{'params': net.parameters()}]
        optim = torch.optim.Adam(params_to_mri, lr=lr, weight_decay=weight_decay)

        if self.model == 'vit_scheduler':
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optim, lr_lambda=lambda epoch: 0.95 ** epoch)

        st_epoch = 0

        transform = transforms.Compose([Normalization(normal=True), RandomFlip(), ToTensor()])
        dataset_train = Dataset(data_dir=dir_data, transforms=transform, dataset='training', model=self.model)
        dataset_val = Dataset(data_dir=dir_data, transforms=transform, dataset='validation', model=self.model)
        # for test image
        transform = transforms.Compose([Resize((224, 224)), Normalization(normal=True), RandomFlip(), ToTensor()])
        dataset_test = TestDataset(data_dir=dir_data, transforms=transform)
        loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_worker,
                                  pin_memory=True, drop_last=True)
        loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=num_worker,
                                pin_memory=True, drop_last=True)
        loader_test = DataLoader(dataset_test, batch_size=8, shuffle=True, num_workers=num_worker,
                                 pin_memory=True, drop_last=True)

        loss_arr = []
        loss_arr_val = []
        train_rmse_arr = []
        val_rmse_arr = []
        early_stopping = EarlyStopping(patience=self.patience, verbose=True)

        for epoch in range(st_epoch + 1, num_epoch + 1):
            net.train()
            if self.model == 'ensemble':
                net_vit.train()
            elif self.model == 'use_meta':
                model.train()
            for batch_idx, data in enumerate(tqdm(loader_train), start=1):
                img = data['img'].to(device)
                label = data['label'].to(device)
                meta = data['meta'].to(device)
                output = net(img)
                if self.model == 'ensemble':
                    output_vit = net_vit(img)
                    output = (output_vit + output) / 2
                elif self.model == 'use_meta':
                    output_vit = net_vit(img)
                    output = model(torch.cat((output, output_vit, meta), dim=1))
                    # output = model(torch.cat((output, meta), dim=1))

                loss = fn_loss_MSE(output, label).to(device)
                train_rmse = RMSELoss(output, label).to(device)

                optim.zero_grad()
                loss.backward()
                optim.step()
                loss_arr.append(loss.item())
                train_rmse_arr.append(train_rmse.item())
                del output

            train_loss = np.mean(loss_arr)
            train_rmse = np.mean(train_rmse_arr)
            writer_train.add_scalar(f'loss/train_MSE_loss', train_loss, epoch)
            writer_train.add_scalar(f'loss/train_RMSE', train_rmse, epoch)

            net.eval()
            if self.model == 'ensemble':
                net_vit.eval()
            elif self.model == 'use_meta':
                model.eval()

            with torch.no_grad():

                for batch_idx, data in enumerate(tqdm(loader_val)):
                    img = data['img'].to(device)
                    label = data['label'].to(device)
                    meta = data['meta'].to(device)
                    output = net(img)
                    if self.model == 'ensemble':
                        output_vit = net_vit(img)
                        output = (output_vit + output) / 2
                    elif self.model == 'use_meta':
                        output_vit = net_vit(img)
                        output = model(torch.cat((output, output_vit, meta), dim=1))
                        # output = model(torch.cat((output, meta), dim=1))

                    loss_val = fn_loss_MSE(output, label).to(device)
                    val_rmse = RMSELoss(output, label).to(device)

                    loss_arr_val.append(loss_val.item())
                    val_rmse_arr.append(val_rmse.item())
                    del output
            if self.model == 'vit_scheduler':
                scheduler.step()

            val_loss = np.mean(loss_arr_val)
            val_rmse = np.mean(val_rmse_arr)
            writer_train.add_scalar(f'loss/val_MSE_loss', val_loss, epoch)
            writer_train.add_scalar(f'loss/val_RMSE', val_rmse, epoch)

            output_list = [epoch, train_loss, val_loss, train_rmse, val_rmse]

            print(output_list)

            early_stopping(val_loss, net)

            if early_stopping.early_stop:
                print("Early stopping")
                break

        save_path = f'{dir_out}/{self.model}/net.pth'
        save(path=save_path, net=net, optim=optim)

        # print("Start testing")
        # with torch.no_grad():
        #     net.eval()
        #     test_output = []
        #     for batch_idx, data in enumerate(tqdm(loader_test), start=1):
        #         img = data['img'].to(device)
        #         id = data['Id']
        #         output = net(img)
        #         test_output.append(to_cpu(output))
        #
        #     test_df = pd.DataFrame({'Id': id, 'Pawpularity': test_output})
        #     test_df.to_csv(f'./{self.dir_ckpt}/{self.model}_test.csv')
        #
        #     print(test_output)

        writer_train.close()
