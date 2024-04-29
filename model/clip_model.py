'''
Author: Zhikai Li luckydogqaq@163.com
Date: 2024-04-12 16:26:45
LastEditors: Zhikai Li luckydogqaq@163.com
LastEditTime: 2024-04-29 19:39:41
FilePath: /clip4sbsr/model/clip_model.py
Description: 

Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
'''

import os
import argparse
from random import sample, randint

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

from model.sketch_model import SketchModel
from model.view_model import MVCNN
from model.classifier import Classifier
from dataset.view_dataset import MultiViewDataSet
from loss.am_softmax import AMSoftMaxLoss
from loss.triplet_center_loss import TripletCenterLoss
import os

from tqdm import tqdm
from pathlib import Path
import yaml
from easydict import EasyDict

from utils.metric import evaluation_metric, cal_cosine_distance

from peft import LoraConfig

import wandb


import torch
import torch.nn as nn
import lightning as L


class Clip4SbsrModel(L.LightningModule):
    def __init__(self, config):
        super(Clip4SbsrModel, self).__init__()
        self.config = config
        self.initialize_model()
        self.initialize_optimizers()

        # self.metrics = {}
        # for stage in ["train", "valid", "test"]:
        #     stage_metrics = self.initialize_metrics(stage)
        #     # Rigister metrics as attributes
        #     for metric_name, metric in stage_metrics.items():
        #         setattr(self, metric_name, metric)
                
        #     self.metrics[stage] = stage_metrics

        self.sketch_features = []
        self.sketch_labels = []
        self.view_features = []
        self.view_labels = []

        self.train_correct = 0
        self.train_total = 0
        self.valid_correct = 0
        self.valid_total = 0
        self.best_value = 0

        self.epoch = 0
        
    # def encode_sketch():
    #     pass

    # def encode_shape():
    #     pass
        
    # def forward():
    #     pass
    
    def initialize_model(self):
        if self.config.prompt.use_prompt:
            self.prompt = nn.Parameter(torch.randn(self.config.prompt.prompt_dim))
            prompt_dim = self.config.prompt.prompt_dim
        else:
            self.prompt = None
            prompt_dim = 0
        
        lora_config = LoraConfig(target_modules=["q_proj", "k_proj"],
                                 r=self.config.lora_rank,
                                 lora_alpha=16,
                                 lora_dropout=0.1)
        
        self.sketch_model = SketchModel(lora_config, self.config.backbone)
        self.view_model = MVCNN(lora_config, self.config.backbone)
        self.classifier = Classifier(self.config.classifier.alph, self.config.classifier.feat_dim + prompt_dim, self.config.classifier.num_classes)
        self.centers = nn.Parameter(torch.randn(self.config.classifier.num_classes, self.config.classifier.num_classes)) 
        
    def initialize_metrics(self, stage):
        return {}
    
    def initialize_optimizers(self):
        param_list = [
            {
                "params": filter(lambda p: p.requires_grad, self.sketch_model.parameters()), 
                "lr": self.config.lr_model
            },
            {
                "params": filter(lambda p: p.requires_grad, self.view_model.parameters()), 
                "lr": self.config.lr_model
            },
            {
                "params": self.classifier.parameters(), 
                "lr": self.config.lr_model * 10
            }]
        
        if self.config.prompt.use_prompt:
            param_list.append({
                "params": [self.prompt], 
                "lr": self.config.prompt.lr_prompt
            })

        # centers
        param_list.append({
               "params": [self.centers], 
               "lr": self.config.lr_model
           })

        self.optimizer = optim.SGD(param_list, lr=self.config.lr_model, momentum=0.9, weight_decay=2e-5)

        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100, last_epoch=-1)

    def check_save_condition(self, now_value: float, mode: str, save_info: dict = None) -> None:
        """
        Check whether to save model. If save_path is not None and now_value is the best, save model.
        Args:
            now_value: Current metric value
            mode: "min" or "max", meaning whether the lower the better or the higher the better
            save_info: Other info to save
        """

        assert mode in ["min", "max"], "mode should be 'min' or 'max'"

        if self.config.save_path is not None:
            dir = os.path.dirname(self.config.save_path)
            os.makedirs(dir, exist_ok=True)
            # save the best checkpoint
            best_value = getattr(self, f"best_value", None)
            if best_value:
                if mode == "min" and now_value < best_value or mode == "max" and now_value > best_value:
                    setattr(self, "best_value", now_value)
                    self.save_checkpoint()

            else:
                setattr(self, "best_value", now_value)
                self.save_checkpoint()

    def save_checkpoint(self):
        if not Path(self.config.save_path).exists():
            Path(self.config.save_path).mkdir(parents=True, exist_ok=True)

        torch.save(self.classifier.state_dict(), Path(self.config.save_path) / 'mlp_layer.pth')
        self.sketch_model.save(Path(self.config.save_path) / 'sketch_lora')
        self.view_model.save(Path(self.config.save_path) / 'view_lora')
        if self.config.prompt.use_prompt: 
            print(f"save epoch {self.epoch} checkpoint!")
            torch.save(self.prompt.detach(), Path(self.config.save_path) / 'prompt.pth')
        
    def load_checkpoint(self):
        self.sketch_model.load(Path(self.config.save_path) /'sketch_lora')
        self.view_model.load(Path(self.config.save_path) / 'view_lora')
        self.classifier.load_state_dict(torch.load(Path(self.config.save_path) / 'mlp_layer.pth'))
        if self.config.prompt.use_prompt:
            self.prompt = nn.Parameter(torch.load(Path(self.config.save_path) / 'prompt.pth'))

  
    def loss_func(self, inputs, targets):
        criterion_am = AMSoftMaxLoss()
        cls_loss = criterion_am(inputs, targets)
        criterion_tcl = TripletCenterLoss()
        tcl_loss = criterion_tcl(inputs, targets, self.centers)
        loss = cls_loss + tcl_loss

        return loss


    def on_train_epoch_start(self):
        return

    def training_step(self, batch, batch_idx):
        sketch_batch, view_batch  = batch
        sketch_datas, sketch_labels = sketch_batch
        view_datas, view_labels = view_batch
        view_datas = torch.stack(view_datas, axis=1)

        sketch_features = self.sketch_model.forward(sketch_datas)
        view_features = self.view_model.forward(view_datas)

        concat_feature = torch.cat((sketch_features, view_features), dim=0)
        concat_labels = torch.cat((sketch_labels, view_labels), dim=0) # (batch_size, )
        logits = self.classifier.forward(concat_feature, mode='train', prompt=self.prompt) # (batch_size, num_classes=133)

        loss = self.loss_func(logits, concat_labels)

        _, predicted = torch.max(logits.data, 1)
        correct = (predicted == concat_labels).sum()
        # acc = correct.item() / concat_labels.size(0)

        self.train_correct += correct
        self.train_total += concat_labels.size(0)
 
        # self.log_dict({"train_loss": loss, "train_acc": acc}, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return loss
    
    def on_train_epoch_end(self):
        self.epoch += 1
        acc = self.train_correct / self.train_total
        self.train_correct, self.train_total = 0, 0
        self.log("train accuracy", acc, prog_bar=False, logger=True)
        # self.check_save_condition(acc, mode="max")
    
    def validation_step(self, batch, batch_idx):
        sketch_batch, view_batch  = batch
        sketch_datas, sketch_labels = sketch_batch
        view_datas, view_labels = view_batch
        view_datas = torch.stack(view_datas, axis=1)

        sketch_features = self.sketch_model.forward(sketch_datas)
        view_features = self.view_model.forward(view_datas)

        concat_feature = torch.cat((sketch_features, view_features), dim=0)
        concat_labels = torch.cat((sketch_labels, view_labels), dim=0) # (batch_size, )
        logits = self.classifier.forward(concat_feature, mode='train', prompt=self.prompt) # (batch_size, num_classes=133)

        loss = self.loss_func(logits, concat_labels)

        _, predicted = torch.max(logits.data, 1)
        correct = (predicted == concat_labels).sum()
        # acc = correct.item() / concat_labels.size(0)
        
        self.valid_correct += correct
        self.valid_total += concat_labels.size(0)

        self.log("valid_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        # self.log_dict({"valid_loss": loss, "valid_acc": acc}, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return loss

    def on_validation_epoch_end(self):
        # pred = self.validation_step_outputs
        # self.validation_step_outputs.clear()
        acc = self.valid_correct / self.valid_total
        self.valid_correct, self.valid_total = 0, 0
        self.log("vliad accuracy", acc, prog_bar=False, logger=True)
        self.check_save_condition(acc, mode="max")


    def test_step(self, batch, batch_idx):
        sketch_batch, view_batch  = batch
        sketch_datas, sketch_labels = sketch_batch
        view_datas, view_labels = view_batch
        view_datas = torch.stack(view_datas, axis=1)

        sketch_feature_batch = self.extract_feature("sketch", sketch_datas)
        sketch_label_batch = sketch_labels.detach().cpu().clone().numpy()

        view_feature_batch = self.extract_feature("view", view_datas)
        view_label_batch = view_labels.detach().cpu().clone().numpy()


        self.sketch_features.append(sketch_feature_batch)
        self.sketch_labels.append(sketch_label_batch)

        self.view_features.append(view_feature_batch)
        self.view_labels.append(view_label_batch)


    def on_test_epoch_end(self):
        sketch_features = np.concatenate((self.sketch_features),axis=0)
        view_features = np.concatenate((self.view_features),axis=0)

        sketch_labels = np.concatenate((self.sketch_labels),axis=0)
        view_labels = np.concatenate((self.view_labels),axis=0)

        feature_data = {
            'sketch_features': sketch_features, 
            'sketch_labels': sketch_labels,
            'view_features': view_features, 
            'view_labels': view_labels
        }
        torch.save(feature_data, Path(self.config.save_path) / 'feature.pt' )

        distance_matrix = cal_cosine_distance(sketch_features, view_features)
        Av_NN, Av_FT, Av_ST, Av_E, Av_DCG, Av_Precision = evaluation_metric(distance_matrix, sketch_labels, view_labels, 'cosine')
        log_dict = {
            'Av_NN': Av_NN.mean(),
            'Av_FT': Av_FT.mean(),
            'Av_ST': Av_ST.mean(),
            'Av_E': Av_E.mean(),
            'Av_DCG': Av_DCG.mean(),
            'Av_Precision': Av_Precision.mean()
        }

        self.log_dict(log_dict, logger=True)
        
    
    def configure_optimizers(self):
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": self.scheduler
            }
    
    
    '''
    description: 
    param {*} self: 
    param {*} modal: "sketch" or "view"
    param {*} data: a batch of datas
    return {*} features of datas
    '''
    def extract_feature(self, modal, datas):
        if modal == "sketch":
            output = self.sketch_model.forward(datas)
        if modal == "view":
            output = self.view_model.forward(datas)
        mu_embeddings= self.classifier.forward(output,mode="test",prompt=self.prompt)

        feature_batch = nn.functional.normalize(mu_embeddings, dim=1)
        feature_batch_numpy = feature_batch.detach().cpu().clone().numpy()

        return feature_batch_numpy
                