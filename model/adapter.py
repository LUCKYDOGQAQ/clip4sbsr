import torch
import torch.nn as nn
from torch.nn import functional as F


class Adapter(nn.Module):
    def __init__(self, c_in,ratio=0.2, reduction=4,bias=False):
        super(Adapter, self).__init__()
        self.ratio=torch.tensor(ratio,requires_grad=True)
        self.bias=bias
        # self.fc = nn.Sequential(
        #     nn.Linear(c_in, c_in // reduction, bias=False),
        #     nn.ReLU(),
        #     nn.Linear(c_in // reduction, c_in, bias=False),
        #     nn.ReLU()
        # )
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=self.bias),
            #nn.BatchNorm1d(c_in // reduction, eps=2e-5),
            nn.ReLU(),
            nn.Linear(c_in // reduction, c_in, bias=self.bias),
            #nn.BatchNorm1d(c_in, eps=2e-5),
            nn.ReLU(),
        )
        #torch.nn.init.kaiming_uniform_(self.parameters())

    def forward(self, x):
        #print(self.ratio)
        x1 = self.fc(x)
        x=self.ratio*x1+(1-self.ratio)*x
        #x = x1
        x=x/ x.norm(dim=-1, keepdim=True)
        return x