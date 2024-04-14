'''
Author: Zhikai Li luckydogqaq@163.com
Date: 2024-04-14 15:57:13
LastEditors: Zhikai Li luckydogqaq@163.com
LastEditTime: 2024-04-14 16:13:24
FilePath: /clip4sbsr/dataset/clip_dataset.py
Description: 

Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
'''
from torch.utils.data.dataset import Dataset

from dataset.view_dataset import MultiViewDataSet
from dataset.sketch_dataset import SketchDataSet

class Clip4SbsrDataset(Dataset):
    def __init__(self, sketch_datadir, sketch_transforms, view_datadir, view_transform) -> None:
        self.sketch_dataset = SketchDataSet(sketch_datadir, sketch_transforms)
        self.view_dataset = MultiViewDataSet(view_datadir, view_transform)

    def __getitem__(self, index):
        sketch_item = self.sketch_dataset[index % len(self.sketch_dataset)]
        view_item = self.view_dataset[index % len(self.view_dataset)]
        return (sketch_item, view_item)
    
    def __len__(self):
        return len(self.sketch_dataset) if len(self.sketch_dataset) >= len(self.view_dataset) else len(self.view_dataset)