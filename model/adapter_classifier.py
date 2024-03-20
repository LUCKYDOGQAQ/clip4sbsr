import torch
import torch.nn as nn
from torch.nn import functional as F

class Classifier(nn.Module):
    def __init__(self, c_in,layers=1, reduction=4):
        super(Classifier, self).__init__()
        self.layers=layers
        self.fc_layers=[]
        for i in range(self.layers):
            self.fc_layers.append(nn.Sequential(
                nn.Linear(c_in, c_in * reduction),
                nn.BatchNorm1d(c_in * reduction, eps=2e-5),
                nn.ReLU(),
                nn.Linear(c_in * reduction, c_in),
                nn.BatchNorm1d(c_in, eps=2e-5),
                nn.ReLU(),
            ))
        self.fc_layers=nn.ModuleList(self.fc_layers)

    def forward(self, x):
        for i in range(self.layers):
            x = self.fc_layers[i](x)
        # x=self.fc2(x)
        # x = self.fc3(x)
        # x = self.fc4(x)
        # x = self.fc5(x)
        # x = self.fc6(x)
        x = x / x.norm(dim=-1, keepdim=True)
        return x