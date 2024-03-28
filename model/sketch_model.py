# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 17:21:12 2018

@author: shirhe-lyh
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
from transformers import CLIPVisionModelWithProjection


class SketchModel(nn.Module):
    """ definition."""

    def __init__(self,lora_config=None,backbone="/lizhikai/workspace/clip4sbsr/hf_model/models--openai--clip-vit-base-patch32"):
        super(SketchModel, self).__init__()
        self.model=CLIPVisionModelWithProjection.from_pretrained(backbone)
        if lora_config is not None:
            self.model.add_adapter(lora_config, adapter_name="sketch_adapter")

    def forward(self, x):
        """
        Args:
            x: input a batch of image

        Returns:
            feature: Extracted features,feature matrix with shape (batch_size, feat_dim),which to be passed
                to the Center Loss

            logits:  prediction tensors to be passed to the Cross Entropy Loss
        """
        feature = self.model(x).image_embeds

        return feature

    def save(self, path):
        self.model.save_pretrained(path)

    def load(self,path):
        self.model.load_adapter(path, adapter_name="sketch_lora")
        self.model.set_adapter("sketch_lora")




