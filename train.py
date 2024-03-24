# -*- coding: utf-8 -*-
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
from reader.view_dataset_reader import MultiViewDataSet
from loss.am_softmax import AMSoftMaxLoss
import os

from tqdm import tqdm
from pathlib import Path



from peft import LoraConfig

parser = argparse.ArgumentParser("Sketch_View Modality")
# dataset
parser.add_argument('--sketch-datadir', type=str, default='../SHREC14_ZS2/14_sketch_train_picture')
parser.add_argument('--view-datadir', type=str, default='../SHREC14_ZS2/14_view_render_train_img')
parser.add_argument('--workers', default=6, type=int,
                    help="number of data loading workers (default: 0)")
# optimization
parser.add_argument('--sketch-batch-size', type=int, default=256)
parser.add_argument('--view-batch-size', type=int, default=256)
parser.add_argument('--num-classes', type=int, default=133)
parser.add_argument('--lr-model', type=float, default=1e-3, help="learning rate for model")
parser.add_argument('--max-epoch', type=int, default=100)
parser.add_argument('--stepsize', type=int, default=10)
parser.add_argument('--gamma', type=float, default=0.9, help="learning rate decay")
parser.add_argument('--feat-dim', type=int, default=512, help="feature size")
parser.add_argument('--alph', type=float, default=12, help="L2 alph")
parser.add_argument('--lora-rank', type=int, default=32, help="rank")
# model
parser.add_argument('--model', type=str, default="openai/clip-vit-base-patch32")
# misc
parser.add_argument('--print-freq', type=int, default=10)
parser.add_argument('--save-model-freq', type=int, default=10)
parser.add_argument('--gpu', type=str, default='0,1,2,3')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--model-dir', type=str, default='./saved_model/')
parser.add_argument('--count', type=int, default=0)

args = parser.parse_args()


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
    sketch_dataloaders = DataLoader(sketch_data, batch_size=args.sketch_batch_size, shuffle=True,
                                    num_workers=args.workers)

    view_data = MultiViewDataSet(view_datadir, transform=view_transform)
    view_dataloaders = DataLoader(view_data, batch_size=args.view_batch_size, shuffle=True, num_workers=args.workers)

    return sketch_dataloaders, view_dataloaders


def train(sketch_model, view_model, classifier, criterion_am,
          optimizer_model, sketch_dataloader, view_dataloader, use_gpu):
    sketch_model.train()
    view_model.train()
    classifier.train()

    total = 0.0
    correct = 0.0

    view_size = len(view_dataloader)
    sketch_size = len(sketch_dataloader)

    sketch_dataloader_iter = iter(sketch_dataloader)
    view_dataloader_iter = iter(view_dataloader)

    for iteration, batch_idx in tqdm(enumerate(range(max(view_size, sketch_size)))):
        if iteration==1:
            break
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
            sketch_data, sketch_labels, view_data, view_labels = sketch_data.to("mps"), sketch_labels.to("mps"), \
                                                                 view_data.to("mps"), view_labels.to("mps")


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
        if (batch_idx + 1) % args.print_freq == 0:
            print("Iter [%d/%d] Total Loss: %.4f" % (batch_idx + 1, max(view_size, sketch_size), loss.item()))
            print("\tAverage Accuracy: %.4f" % (avg_acc))

        args.count += 1

        # writer.add_scalar("Loss", loss.item(), args.count)
        # writer.add_scalar("average accuracy", avg_acc, args.count)

    return avg_acc


def main():
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
    os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
    use_gpu = torch.cuda.is_available()
    if torch.cuda.is_available(): device = "cuda"
    elif torch.backends.mps.is_available: device = "mps"
    else: device = "cpu"

    if use_gpu:
        print("Currently using GPU: {}".format(args.gpu))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU")

    print("Creating model: {}".format(args.model))

    lora_config=LoraConfig(target_modules=["q_proj", "k_proj"],
                           r=args.lora_rank,
                           lora_alpha=16,
                           lora_dropout=0.1)

    sketch_model = SketchModel(lora_config=lora_config,backbone=args.model)
    view_model = MVCNN(lora_config=lora_config,backbone=args.model)
    classifier = Classifier(args.alph, args.feat_dim, args.num_classes)



    sketch_model = sketch_model.to("mps")
    view_model = view_model.to("mps")
    classifier = classifier.to("mps")


    # Cross Entropy Loss and Center Loss
    criterion_am = AMSoftMaxLoss()
    optimizer_model = torch.optim.SGD([{"params": filter(lambda p: p.requires_grad, sketch_model.parameters()), "lr": args.lr_model},
                                       {"params": filter(lambda p: p.requires_grad, view_model.parameters()), "lr": args.lr_model},
                                       {"params": classifier.parameters(), "lr": args.lr_model * 10}],
                                      lr=args.lr_model, momentum=0.9, weight_decay=2e-5)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer_model, T_max=args.max_epoch, last_epoch=-1)

    sketch_trainloader, view_trainloader = get_data(args.sketch_datadir, args.view_datadir)
    for epoch in range(args.max_epoch):
        print("==> Epoch {}/{}".format(epoch + 1, args.max_epoch))
        print("++++++++++++++++++++++++++")
        # save model

        train(sketch_model, view_model, classifier, criterion_am,
              optimizer_model, sketch_trainloader, view_trainloader, use_gpu)


        model_save_path = Path(args.model_dir +'Epoch'+str(epoch))
        if not model_save_path.exists():
            # os.makedirs(model_save_path)
            model_save_path.mkdir(parents=True, exist_ok=True)
        torch.save(classifier.state_dict(),
                   model_save_path / 'mlp_layer.pth')
        sketch_model.save(model_save_path / 'sketch_lora')
        view_model.save(model_save_path / 'view_lora')

        if args.stepsize > 0: scheduler.step()


if __name__ == '__main__':
    main()