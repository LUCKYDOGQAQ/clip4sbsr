# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 17:21:12 2018

@author: shirhe-lyh
"""
import numpy as np
import random

import torch
import torch.nn as nn
from torchvision import models
from transformers import CLIPVisionModelWithProjection

class MVCNN(nn.Module):
    """definition."""

    def __init__(self,lora_config=None,backbone="/lizhikai/workspace/clip4sbsr/hf_model/models--openai--clip-vit-base-patch32"):
        super(MVCNN, self).__init__()
        self.model=CLIPVisionModelWithProjection.from_pretrained(backbone)
        if lora_config is not None:
            self.model.add_adapter(lora_config, adapter_name="view_adapter")
        
        if torch.cuda.is_available(): self.device = "cuda"
        elif torch.backends.mps.is_available: self.device = "mps"
        else: self.device = "cpu"


    def forward(self, x, prompt=None):
        """
        Args:
            x: input a batch of image

        Returns:
            pooled_view: Extracted features, maxpooling of multiple features of 12 view_images of 3D model

            logits:  prediction tensors to be passed to the Cross Entropy Loss
        """
        """
        x = x.transpose(0, 1)
        rand = random.sample(range(1,12),6)
        x = x[rand]
        #print(type(x))
        #print(x.shape)

        view_pool = []

        for v in x:
            v = v.type(torch.cuda.FloatTensor)

            feature = self.model(v)
            feature = feature.view(feature.size(0), self.feature_size)  #
            feature = feature.detach().cpu().clone().numpy()
            view_pool.append(feature)

        #rand = random.randint(0, 12)
        #view_pool = view_pool[rand]
        view_pool = np.array(view_pool)
        view_pool1 = torch.from_numpy(view_pool)
        #print(view_pool1.size())
        #pooled_view = view_pool[0]
        pooled_view = torch.mean(view_pool1,dim = 0)
        #print(pooled_view.size())
        #for i in range(1, len(view_pool)):
            #pooled_view = torch.max(pooled_view, view_pool[i])  # max_pooling
        #print(pooled_view.size())
        pooled_view = pooled_view.type(torch.cuda.FloatTensor)
        #logits = self.fc(pooled_view)
        #pooled_view = self.layer1(pooled_view)
        #feature = self.layer2(feature)
        #feature = self.fc1(feature)"""

        batch_size = x.shape[0]
        x = x.transpose(0, 1)
        view_pool = []

        for v in x:
            v = v.type(torch.FloatTensor).to(self.device)
            # feature = self.model(v).image_embeds

            v = self.model.vision_model.embeddings(v)

            if prompt is not None:
                prompt = prompt.expand(batch_size, -1, -1) # shape = [batch_size, num_prompt, feat_dim]
                v = torch.cat([v, prompt], dim=1) # # shape = [batch_size, num_patch+num_prompt , feat_dim]

            v = self.model.vision_model.pre_layrnorm(v)
            last_hidden_state = self.model.vision_model.encoder(v)['last_hidden_state']
            pooled_output = last_hidden_state[:, 0, :]
            pooled_output = self.model.vision_model.post_layernorm(pooled_output)
            feature = self.model.visual_projection(pooled_output)

            feature=feature.unsqueeze(0)
            view_pool.append(feature)

        pooled_view = view_pool[0]
        for i in range(1, len(view_pool)):
            pooled_view = torch.cat((pooled_view, view_pool[i]),dim=0)  #
        pooled_view = torch.mean(pooled_view,dim=0)  #
        return pooled_view

    def save(self, path):
        self.model.save_pretrained(path)

    def load(self,path):
        self.model.load_adapter(path,adapter_name="view_lora")
        self.model.set_adapter("view_lora")