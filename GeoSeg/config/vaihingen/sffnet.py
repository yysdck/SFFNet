from torch.utils.data import DataLoader
from GeoSeg.geoseg.losses.useful_loss import Loss
from GeoSeg.geoseg.datasets.vaihingen_dataset import *
from catalyst.contrib.nn.optimizers import Lookahead
import catalyst.utils as utils
from GeoSeg.geoseg.models.SFFNet.SFFNet import SFFNet

# training hparam
max_epoch = 105
ignore_index = len(CLASSES)
train_batch_size = 2
val_batch_size = 2
lr = 6e-4
weight_decay = 0.01
backbone_lr = 6e-5
backbone_weight_decay = 0.01
accumulate_n = 1
num_classes = len(CLASSES)
classes = CLASSES

weights_name = "sffnet-convnext-512-crop-ms-e105"
weights_path = "model_weights/vaihingen/{}".format(weights_name)
test_weights_name = "sffnet-convnext-512-crop-ms-e105"
log_name = 'vaihingen/{}'.format(weights_name)
monitor = 'val_F1'
monitor_mode = 'max'
save_top_k = 3
save_last = True
check_val_every_n_epoch = 1
gpus = [0]
strategy = None
pretrained_ckpt_path = None
resume_ckpt_path = None



# define the loss
trainloss = Loss(ignore_index=ignore_index)
valloss = Loss(ignore_index=ignore_index)
use_aux_loss = False

#  define the networ
net = SFFNet(num_classes=num_classes)
# define the dataloader

train_dataset = VaihingenDataset(data_root='data/vaihingen/train', mode='train',
                                 mosaic_ratio=0.25, transform=train_aug)

val_dataset = VaihingenDataset(transform=val_aug)
test_dataset = VaihingenDataset(data_root='data/vaihingen/test',
                                transform=val_aug)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          num_workers=2,
                          pin_memory=True,
                          shuffle=True,
                          drop_last=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=2,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False)
# define the optimizer
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = utils.torch.process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)