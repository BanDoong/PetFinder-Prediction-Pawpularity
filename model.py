import torch
import torch.nn as nn


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class EfficientHybridviT(nn.Module):
    def __init__(self, base):
        super(EfficientHybridviT, self).__init__()
        from pytorch_pretrained_vit import ViT

        self._conv_stem_ex = base._conv_stem
        self._bn0_ex = base._bn0
        self._blocks_0_ex = base._blocks[0]
        self._blocks_1_ex = base._blocks[1]
        self._blocks_2_ex = base._blocks[2]
        self.vit = ViT('B_16', pretrained=True, num_classes=1)
        self.conv = nn.Conv2d(in_channels=24, out_channels=3, kernel_size=3, padding='same')

    def forward(self, x):
        x = self._conv_stem_ex(x)
        x = self._bn0_ex(x)
        x = self._blocks_0_ex(x)
        x = self._blocks_1_ex(x)
        x = self._blocks_2_ex(x)
        x = self.conv(x)
        return self.vit(x)


class EfficientHybridviT_2(nn.Module):
    def __init__(self, base):
        super(EfficientHybridviT_2, self).__init__()
        from pytorch_pretrained_vit import ViT

        self._conv_stem_ex = base._conv_stem
        self._bn0_ex = base._bn0
        self._blocks_ex = base._blocks
        self._conv_head_ex = base._conv_head
        self._blocks_0_ex = base._blocks[0]
        self._blocks_1_ex = base._blocks[1]
        self._blocks_2_ex = base._blocks[2]
        self._blocks_3_ex = base._blocks[3]
        self._blocks_4_ex = base._blocks[4]
        self._blocks_5_ex = base._blocks[5]
        self._blocks_6_ex = base._blocks[6]
        self._blocks_7_ex = base._blocks[7]
        self._blocks_8_ex = base._blocks[8]
        self._blocks_9_ex = base._blocks[9]
        self._blocks_10_ex = base._blocks[10]
        self._blocks_11_ex = base._blocks[11]
        self._blocks_12_ex = base._blocks[12]
        self._blocks_13_ex = base._blocks[13]
        self._blocks_14_ex = base._blocks[14]
        self._blocks_15_ex = base._blocks[15]
        self.vit = ViT('B_16', pretrained=True, num_classes=1000)
        self.conv = nn.Conv2d(in_channels=24, out_channels=3, kernel_size=3, padding='same')
        self.fc = nn.Linear(in_features=2280, out_features=1)
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)

    def forward(self, x):
        x = self._conv_stem_ex(x)
        x = self._bn0_ex(x)
        x = self._blocks_0_ex(x)
        x = self._blocks_1_ex(x)
        x = self._blocks_2_ex(x)
        x1 = self.conv(x)
        x1 = self.vit(x1)

        x2 = self._blocks_3_ex(x)
        x2 = self._blocks_4_ex(x2)
        x2 = self._blocks_5_ex(x2)
        x2 = self._blocks_6_ex(x2)
        x2 = self._blocks_7_ex(x2)
        x2 = self._blocks_8_ex(x2)
        x2 = self._blocks_9_ex(x2)
        x2 = self._blocks_10_ex(x2)
        x2 = self._blocks_11_ex(x2)
        x2 = self._blocks_12_ex(x2)
        x2 = self._blocks_13_ex(x2)
        x2 = self._blocks_14_ex(x2)
        x2 = self._blocks_15_ex(x2)
        x2 = self._conv_head_ex(x2)

        x2 = self.pool(x2)
        x2 = x2.squeeze()

        x = torch.cat((x1, x2), dim=1)
        x = self.fc(x)
        return x


class EfficientHybridSwin(nn.Module):
    def __init__(self, base):
        super(EfficientHybridSwin, self).__init__()
        # from swin_trans import SwinTransformer

        self._conv_stem_ex = base._conv_stem
        self._bn0_ex = base._bn0
        self._blocks_ex = base._blocks
        self._conv_head_ex = base._conv_head
        self._blocks_0_ex = base._blocks[0]
        self._blocks_1_ex = base._blocks[1]
        self._blocks_2_ex = base._blocks[2]
        self._blocks_3_ex = base._blocks[3]
        self._blocks_4_ex = base._blocks[4]
        self._blocks_5_ex = base._blocks[5]
        self._blocks_6_ex = base._blocks[6]
        self._blocks_7_ex = base._blocks[7]
        self._blocks_8_ex = base._blocks[8]
        self._blocks_9_ex = base._blocks[9]
        self._blocks_10_ex = base._blocks[10]
        self._blocks_11_ex = base._blocks[11]
        self._blocks_12_ex = base._blocks[12]
        self._blocks_13_ex = base._blocks[13]
        self._blocks_14_ex = base._blocks[14]
        self._blocks_15_ex = base._blocks[15]
        import timm
        self.swin = timm.create_model('swin_base_patch4_window7_224_in22k', pretrained=True)
        n_fts = self.swin.head.in_features
        self.swin.head = nn.Linear(n_fts, out_features=1000, bias=True)

        self.conv = nn.Conv2d(in_channels=24, out_channels=3, kernel_size=3, padding='same')
        self.fc = nn.Linear(in_features=2280, out_features=1)
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)

    def forward(self, x):
        x = self._conv_stem_ex(x)
        x = self._bn0_ex(x)
        x = self._blocks_0_ex(x)
        x = self._blocks_1_ex(x)
        x = self._blocks_2_ex(x)
        x1 = self.conv(x)
        x1 = self.swin(x1)

        x2 = self._blocks_3_ex(x)
        x2 = self._blocks_4_ex(x2)
        x2 = self._blocks_5_ex(x2)
        x2 = self._blocks_6_ex(x2)
        x2 = self._blocks_7_ex(x2)
        x2 = self._blocks_8_ex(x2)
        x2 = self._blocks_9_ex(x2)
        x2 = self._blocks_10_ex(x2)
        x2 = self._blocks_11_ex(x2)
        x2 = self._blocks_12_ex(x2)
        x2 = self._blocks_13_ex(x2)
        x2 = self._blocks_14_ex(x2)
        x2 = self._blocks_15_ex(x2)
        x2 = self._conv_head_ex(x2)

        x2 = self.pool(x2)
        x2 = x2.squeeze()

        x = torch.cat((x1, x2), dim=1)
        x = self.fc(x)
        return x


class meta_model(nn.Module):
    def __init__(self):
        super(meta_model, self).__init__()
        self.fc = nn.Linear(36, 1)

    def forward(self, x):
        return self.fc(x)

# class ViT(nn.Module):
#     """
#     Args:
#         name (str): Model name, e.g. 'B_16'
#         pretrained (bool): Load pretrained weights
#         in_channels (int): Number of channels in input data
#         num_classes (int): Number of classes, default 1000
#     References:
#         [1] https://openreview.net/forum?id=YicbFdNTTy
#     """
#
#     def __init__(
#             self,
#             name: Optional[str] = None,
#             pretrained: bool = False,
#             patches: int = 16,
#             dim: int = 768,
#             ff_dim: int = 3072,
#             num_heads: int = 12,
#             num_layers: int = 12,
#             attention_dropout_rate: float = 0.0,
#             dropout_rate: float = 0.1,
#             representation_size: Optional[int] = None,
#             load_repr_layer: bool = False,
#             classifier: str = 'token',
#             positional_embedding: str = '1d',
#             in_channels: int = 3,
#             image_size: Optional[int] = None,
#             num_classes: Optional[int] = None,
#     ):
#         super().__init__()
#
#         # Configuration
#         if name is None:
#             check_msg = 'must specify name of pretrained model'
#             assert not pretrained, check_msg
#             assert not resize_positional_embedding, check_msg
#             if num_classes is None:
#                 num_classes = 1000
#             if image_size is None:
#                 image_size = 384
#         else:  # load pretrained model
#             assert name in PRETRAINED_MODELS.keys(), \
#                 'name should be in: ' + ', '.join(PRETRAINED_MODELS.keys())
#             config = PRETRAINED_MODELS[name]['config']
#             patches = config['patches']
#             dim = config['dim']
#             ff_dim = config['ff_dim']
#             num_heads = config['num_heads']
#             num_layers = config['num_layers']
#             attention_dropout_rate = config['attention_dropout_rate']
#             dropout_rate = config['dropout_rate']
#             representation_size = config['representation_size']
#             classifier = config['classifier']
#             if image_size is None:
#                 image_size = PRETRAINED_MODELS[name]['image_size']
#             if num_classes is None:
#                 num_classes = PRETRAINED_MODELS[name]['num_classes']
#         self.image_size = image_size
#
#         # Image and patch sizes
#         h, w = as_tuple(image_size)  # image sizes
#         fh, fw = as_tuple(patches)  # patch sizes
#         gh, gw = h // fh, w // fw  # number of patches
#         seq_len = gh * gw
#
#         # Patch embedding
#         self.patch_embedding = nn.Conv2d(in_channels, dim, kernel_size=(fh, fw), stride=(fh, fw))
#
#         # Class token
#         if classifier == 'token':
#             self.class_token = nn.Parameter(torch.zeros(1, 1, dim))
#             seq_len += 1
#
#         # Positional embedding
#         if positional_embedding.lower() == '1d':
#             self.positional_embedding = PositionalEmbedding1D(seq_len, dim)
#         else:
#             raise NotImplementedError()
#
#         # Transformer
#         self.transformer = Transformer(num_layers=num_layers, dim=dim, num_heads=num_heads,
#                                        ff_dim=ff_dim, dropout=dropout_rate)
#
#         # Representation layer
#         if representation_size and load_repr_layer:
#             self.pre_logits = nn.Linear(dim, representation_size)
#             pre_logits_size = representation_size
#         else:
#             pre_logits_size = dim
#
#         # Classifier head
#         self.norm = nn.LayerNorm(pre_logits_size, eps=1e-6)
#         self.fc = nn.Linear(pre_logits_size, num_classes)
#
#         # Initialize weights
#         self.init_weights()
#
#         # Load pretrained model
#         if pretrained:
#             pretrained_num_channels = 3
#             pretrained_num_classes = PRETRAINED_MODELS[name]['num_classes']
#             pretrained_image_size = PRETRAINED_MODELS[name]['image_size']
#             load_pretrained_weights(
#                 self, name,
#                 load_first_conv=(in_channels == pretrained_num_channels),
#                 load_fc=(num_classes == pretrained_num_classes),
#                 load_repr_layer=load_repr_layer,
#                 resize_positional_embedding=(image_size != pretrained_image_size),
#             )
#
#     @torch.no_grad()
#     def init_weights(self):
#         def _init(m):
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(
#                     m.weight)  # _trunc_normal(m.weight, std=0.02)  # from .initialization import _trunc_normal
#                 if hasattr(m, 'bias') and m.bias is not None:
#                     nn.init.normal_(m.bias, std=1e-6)  # nn.init.constant(m.bias, 0)
#
#         self.apply(_init)
#         nn.init.constant_(self.fc.weight, 0)
#         nn.init.constant_(self.fc.bias, 0)
#         nn.init.normal_(self.positional_embedding.pos_embedding,
#                         std=0.02)  # _trunc_normal(self.positional_embedding.pos_embedding, std=0.02)
#         nn.init.constant_(self.class_token, 0)
#
#     def forward(self, x):
#         """Breaks image into patches, applies transformer, applies MLP head.
#         Args:
#             x (tensor): `b,c,fh,fw`
#         """
#         b, c, fh, fw = x.shape
#         x = self.patch_embedding(x)  # b,d,gh,gw
#         x = x.flatten(2).transpose(1, 2)  # b,gh*gw,d
#         if hasattr(self, 'class_token'):
#             x = torch.cat((self.class_token.expand(b, -1, -1), x), dim=1)  # b,gh*gw+1,d
#         if hasattr(self, 'positional_embedding'):
#             x = self.positional_embedding(x)  # b,gh*gw+1,d
#         x = self.transformer(x)  # b,gh*gw+1,d
#         if hasattr(self, 'pre_logits'):
#             x = self.pre_logits(x)
#             x = torch.tanh(x)
#         if hasattr(self, 'fc'):
#             x = self.norm(x)[:, 0]  # b,d
#             x = self.fc(x)  # b,num_classes
#         return x

# class ViT_model(pl.LightningModule):
#     def __init__(self, cfg):
#         super().__init__()
#         self.cfg = cfg
#         self.backbone = timm.create_model(cfg.model.name, pretrained=False)
#         self.backbone.load_state_dict(torch.load(cfg.model.weight))
#         for param in self.backbone.parameters():
#             param.requires_grad = False
#         self.backbone.head = nn.Linear(768, 1)
#         self.sigmoid = nn.Sigmoid()
#         self.criterion = nn.BCELoss()
#
#     def forward(self, input):
#         x = self.backbone(input)
#         x = self.sigmoid(x)
#         return x
#
#     def _step(self, batch):
#         img, target = batch
#         pred = self(img)
#         loss = self.criterion(pred, target)
#         return pred, target, loss
#
#     def training_step(self, batch, batch_idx):
#         pred, target, loss = self._step(batch)
#         metric = RMSELoss(pred, target)
#         tensorboard_log = {'train_loss': loss, 'train_rmse': metric}
#         return {'loss': loss, 'rmse': metric, 'log': tensorboard_log}
#
#     def validation_step(self, batch, batch_idx):
#         with torch.no_grad():
#             pred, target, loss = self._step(batch)
#             rmse = RMSELoss(pred * 100.0, target * 100.0)
#         return {'val_loss': loss, 'val_rmse': rmse}
#
#     def validation_epoch_end(self, outputs):
#         avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
#         avg_rmse = torch.stack([x['val_rmse'] for x in outputs]).mean()
#         print(f"Epoch {self.current_epoch} loss:{avg_loss} rmse:{avg_rmse}")
#         self.log('val_rmse', avg_rmse)
#         tensorboard_logs = {'val_loss': avg_loss, 'val_rmse': avg_rmse}
#         return {'val_loss': avg_loss,
#                 'val_rmse': avg_rmse,
#                 'log': tensorboard_logs}
#
#     def configure_optimizers(self):
#         optimizer = eval(self.cfg.optim.name)(self.parameters(), lr=self.cfg.optim.lr)
#         schedule = eval(self.cfg.lr_sched.name)(optimizer=optimizer, **self.cfg.lr_sched.params)
#         return [optimizer], [schedule]
