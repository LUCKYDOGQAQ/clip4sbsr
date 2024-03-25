from torch.utils.data.dataset import Dataset
import os
from PIL import Image
import torch

class ClipDataSet(Dataset):

    def __init__(self, x,y, transform=None, target_transform=None):
        if torch.cuda.is_available(): self.device = "cuda"
        elif torch.backends.mps.is_available: self.device = "mps"
        else: self.device = "cpu"
        self.x = torch.Tensor(x).type(dtype=torch.float).to(self.device)
        self.y = torch.Tensor(y).type(dtype=torch.int64).to(self.device)
        self.transform = transform
        self.target_transform = target_transform

    # Override to give PyTorch access to any image on the dataset
    def __getitem__(self, index):

        return self.x[index], self.y[index]

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.x)