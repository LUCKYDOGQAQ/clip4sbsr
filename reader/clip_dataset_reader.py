from torch.utils.data.dataset import Dataset
import os
from PIL import Image
import torch

class ClipDataSet(Dataset):

    def __init__(self, x,y, transform=None, target_transform=None):
        self.x = torch.Tensor(x).type(dtype=torch.float).to("mps")
        self.y = torch.Tensor(y).type(dtype=torch.int64).to("mps")
        self.transform = transform
        self.target_transform = target_transform

    # Override to give PyTorch access to any image on the dataset
    def __getitem__(self, index):

        return self.x[index], self.y[index]

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.x)