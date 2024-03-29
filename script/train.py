# -*- coding: utf-8 -*-
import sys
sys.path.append('.')

import os
import argparse
from random import sample, randint

import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

from model.sketch_model import SketchModel
from model.view_model import MVCNN
from model.classifier import Classifier
from dataset.view_dataset_reader import MultiViewDataSet
from loss.am_softmax import AMSoftMaxLoss
import os

from tqdm import tqdm
from pathlib import Path
import yaml
from easydict import EasyDict



from peft import LoraConfig

import wandb
# import random

parser = argparse.ArgumentParser("Sketch_View Modality")
parser.add_argument('-c', '--config', help="running configurations", type=str, required=True)
with open(parser.parse_args().config, 'r', encoding='utf-8') as r:
    config = EasyDict(yaml.safe_load(r))


# writer = SummaryWriter()


def get_data(sketch_datadir, view_datadir):
    """Image reading and image augmentation
       Args:
         traindir: path of the traing picture
    """
    image_transforms = transforms.Compose([
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

    sketch_data = datasets.ImageFolder(root=sketch_datadir, transform=image_transforms)
    sketch_dataloaders = DataLoader(sketch_data, batch_size=config.dataset.sketch_batch_size, shuffle=True,
                                    num_workers=config.dataset.workers)

    view_data = MultiViewDataSet(view_datadir, transform=view_transform)
    view_dataloaders = DataLoader(view_data, batch_size=config.dataset.view_batch_size, shuffle=True, num_workers=config.dataset.workers)

    return sketch_dataloaders, view_dataloaders


def train(sketch_model, view_model, classifier, criterion_am,
          optimizer_model, sketch_dataloader, view_dataloader, use_gpu, device):
    sketch_model.train()
    view_model.train()
    classifier.train()

    total = 0.0
    correct = 0.0

    view_size = len(view_dataloader)
    sketch_size = len(sketch_dataloader)

    sketch_dataloader_iter = iter(sketch_dataloader)
    view_dataloader_iter = iter(view_dataloader)

    with tqdm(enumerate(range(max(view_size, sketch_size))), total=max(view_size, sketch_size)) as tbar:
        for iteration, batch_idx in tbar:
            # if iteration==1:
            #     break
            ##################################################################
            # 两个数据集大小不一样，当少的数据集加载完而多的数据集没有加载完的时候，重新加载少的数据集
            if sketch_size > view_size:
                sketch = next(sketch_dataloader_iter)
                try:
                    view = next(view_dataloader_iter)
                except:
                    del view_dataloader_iter
                    view_dataloader_iter = iter(view_dataloader)
                    view = next(view_dataloader_iter)
            else:
                view = next(view_dataloader_iter)
                try:
                    sketch = next(sketch_dataloader_iter)
                except:
                    del sketch_dataloader_iter
                    sketch_dataloader_iter = iter(sketch_dataloader)
                    sketch = next(sketch_dataloader_iter)
            ###################################################################

            sketch_data, sketch_labels = sketch
            view_data, view_labels = view
            view_data = np.stack(view_data, axis=1)
            view_data = torch.from_numpy(view_data)
            if use_gpu:
                sketch_data, sketch_labels, view_data, view_labels = sketch_data.to(device), sketch_labels.to(device), \
                                                                     view_data.to(device), view_labels.to(device)


            sketch_features = sketch_model.forward(sketch_data)
            view_features = view_model.forward(view_data)

            concat_feature = torch.cat((sketch_features, view_features), dim=0)
            concat_labels = torch.cat((sketch_labels, view_labels), dim=0)

            logits = classifier.forward(concat_feature, mode='train')
            cls_loss = criterion_am(logits, concat_labels)
            loss = cls_loss

            _, predicted = torch.max(logits.data, 1)
            total += concat_labels.size(0)
            correct += (predicted == concat_labels).sum()
            avg_acc = correct.item() / total

            loss.backward()
            optimizer_model.step()
            optimizer_model.zero_grad()

            tbar.set_postfix({"loss": loss.item(), "avg_acc": avg_acc})
            wandb.log({"acc": avg_acc, "loss": loss.item()})

    return avg_acc

def load_logger(config):
    # start a new wandb run to track this script
    wandb.init(
        project="CLIP4SBSR",
        name="baseline",
        config=config
    )

def main():

    torch.manual_seed(config.train.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = config.train.gpu
    os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
    os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
    os.environ['WANDB_MODE'] = 'offline'

    load_logger(config)

    use_gpu = torch.cuda.is_available() or torch.backends.mps.is_available()
    if torch.cuda.is_available(): 
        device = "cuda"
    elif torch.backends.mps.is_available(): 
        device = "mps"
    else: 
        device = "cpu"

    print(f"Currently using device: {device}")

    print("Creating model: {}".format(config.model.model))

    lora_config = LoraConfig(target_modules=["q_proj", "k_proj"],
                             r=config.model.lora_rank,
                             lora_alpha=16,
                             lora_dropout=0.1)

    sketch_model = SketchModel(lora_config=lora_config, backbone=config.model.model)
    view_model = MVCNN(lora_config=lora_config, backbone=config.model.model)
    classifier = Classifier(config.model.alph, config.train.feat_dim, config.train.num_classes)

    sketch_model = sketch_model.to(device)
    view_model = view_model.to(device)
    classifier = classifier.to(device)

    # Cross Entropy Loss and Center Loss
    criterion_am = AMSoftMaxLoss()
    optimizer_model = torch.optim.SGD([{"params": filter(lambda p: p.requires_grad, sketch_model.parameters()), "lr": config.train.lr_model},
                                       {"params": filter(lambda p: p.requires_grad, view_model.parameters()), "lr": config.train.lr_model},
                                       {"params": classifier.parameters(), "lr": config.train.lr_model * 10}],
                                      lr=config.train.lr_model, momentum=0.9, weight_decay=2e-5)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer_model, T_max=config.train.max_epoch, last_epoch=-1)

    sketch_trainloader, view_trainloader = get_data(config.dataset.sketch_datadir, config.dataset.view_datadir)
    for epoch in range(config.train.max_epoch):
        print("==> Epoch {}/{}".format(epoch + 1, config.train.max_epoch))
        print("++++++++++++++++++++++++++")
        # Save model
        train(sketch_model, view_model, classifier, criterion_am,
              optimizer_model, sketch_trainloader, view_trainloader, use_gpu, device)

        model_save_path = Path(config.model.ckpt_dir) / f'Epoch{epoch}'
        if not model_save_path.exists():
            model_save_path.mkdir(parents=True, exist_ok=True)
        torch.save(classifier.state_dict(), model_save_path / 'mlp_layer.pth')
        sketch_model.save(model_save_path / 'sketch_lora')
        view_model.save(model_save_path / 'view_lora')

        if config.train.stepsize > 0:
            scheduler.step()

        print("==> Epoch {}/{}".format(epoch + 1, config.train.max_epoch))
        print("++++++++++++++++++++++++++")
        # save model

        avg_acc = train(sketch_model, view_model, classifier, criterion_am,
              optimizer_model, sketch_trainloader, view_trainloader, use_gpu, device)


        model_save_path = Path(config.model.ckpt_dir) / 'exp2' / f'Epoch{epoch}'
        if not model_save_path.exists():
            model_save_path.mkdir(parents=True, exist_ok=True)
        
        torch.save(classifier.state_dict(), model_save_path / 'mlp_layer.pth')
        sketch_model.save(model_save_path / 'sketch_lora')
        view_model.save(model_save_path / 'view_lora')

        if config.train.stepsize > 0: scheduler.step()

        wandb.log({'epoch': epoch, 'accuracy of epoch': avg_acc}, step=epoch)


if __name__ == '__main__':
    main()