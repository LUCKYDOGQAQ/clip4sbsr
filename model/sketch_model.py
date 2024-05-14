'''
Author: Zhikai Li luckydogqaq@163.com
Date: 2024-05-12 20:41:47
LastEditors: Zhikai Li luckydogqaq@163.com
LastEditTime: 2024-05-12 21:13:30
FilePath: /clip4sbsr/model/sketch_model.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
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

    def forward(self, x, prompt=None):
        """
        Args:
            x: input a batch of image

        Returns:
            feature: Extracted features,feature matrix with shape (batch_size, feat_dim),which to be passed
                to the Center Loss

            logits:  prediction tensors to be passed to the Cross Entropy Loss
        """
        # feature = self.model(x).image_embeds

        batch_size = x.shape[0]
        x = self.model.vision_model.embeddings(x)

        if prompt is not None:
            prompt = prompt.expand(batch_size, -1, -1) # shape = [batch_size, num_prompt, feat_dim]
            x = torch.cat([x, prompt], dim=1) # # shape = [batch_size, num_patch+num_prompt , feat_dim]

        x = self.model.vision_model.pre_layrnorm(x)
        last_hidden_state = self.model.vision_model.encoder(x)['last_hidden_state']
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.model.vision_model.post_layernorm(pooled_output)
        feature = self.model.visual_projection(pooled_output)

        return feature

    def save(self, path):
        self.model.save_pretrained(path)

    def load(self,path):
        self.model.load_adapter(path, adapter_name="sketch_lora")
        self.model.set_adapter("sketch_lora")




