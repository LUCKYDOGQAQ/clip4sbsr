'''
Author: Zhikai Li luckydogqaq@163.com
Date: 2024-05-12 21:41:26
LastEditors: Zhikai Li luckydogqaq@163.com
LastEditTime: 2024-05-12 21:41:27
FilePath: /clip4sbsr/script/clip.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import sys
sys.path.append('.')

from model.clip_model import Clip4SbsrModel
from dataset.clip_dataset import Clip4SbsrDataset

import lightning as L
from lightning.pytorch.loggers import WandbLogger

from torchvision import transforms
from torch.utils.data import DataLoader, random_split

import yaml
import argparse
from easydict import EasyDict

import wandb

import os



parser = argparse.ArgumentParser("CLIP for SBSR")
parser.add_argument('-c', '--config', help="running configurations", type=str, required=True)
with open(parser.parse_args().config, 'r', encoding='utf-8') as r:
    config = EasyDict(yaml.safe_load(r))

os.environ['CUDA_VISIBLE_DEVICES'] = config.trainer.gpu
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
os.environ['WANDB_MODE'] = 'offline'

def load_logger(config):
    return WandbLogger(project=config.setting.wandb.project,
                       name=config.setting.wandb.name,
                       config=config)
wandb_logger = load_logger(config)


sketch_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize([0.48145466, 0.4578275, 0.40821073],
                             [0.26862954, 0.26130258, 0.27577711])])  # Imagenet standards

view_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.TrivialAugmentWide(),
    transforms.ToTensor(),
    transforms.Normalize([0.48145466, 0.4578275, 0.40821073],
                         [0.26862954, 0.26130258, 0.27577711])])

clip_model = Clip4SbsrModel(config.model)

clip_dataset = Clip4SbsrDataset(config.dataset.train_sketch_datadir, sketch_transform, config.dataset.train_view_datadir, view_transform)

train_size = int(0.8 * len(clip_dataset))
val_size = len(clip_dataset) - train_size
train_dataset, val_dataset = random_split(clip_dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, 
                             batch_size=config.dataset.batch_size,
                             shuffle=True,
                             num_workers=config.dataset.num_workers,
                            #  sampler=sampler
                             )

val_dataloader = DataLoader(val_dataset, 
                             batch_size=config.dataset.batch_size,
                            #  shuffle=True,
                             num_workers=config.dataset.num_workers,
                            #  sampler=sampler
                             )

trainer = L.Trainer(max_epochs=config.trainer.max_epochs,
                    logger = wandb_logger,
                    accumulate_grad_batches = 1)

trainer.fit(model=clip_model, 
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader)

test_dataset = Clip4SbsrDataset(config.dataset.test_sketch_datadir, sketch_transform, config.dataset.test_view_datadir, view_transform)
test_dataloader = DataLoader(test_dataset, 
                             batch_size=config.dataset.batch_size,
                            #  shuffle=True,
                             num_workers=config.dataset.num_workers,
                            #  sampler=sampler
                             )
clip_model.load_checkpoint()
trainer.test(model=clip_model, 
            dataloaders=test_dataloader)

