import torch
from torch import nn
from torch.nn import functional as F

from models.conv import Conv2d
import numpy as np

class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm2d(cout)
                            )
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)
    

x = np.ones((96,96))
x = torch.FloatTensor(x)
aa=np.ones((1,96,96,15))
aa = np.asarray(aa)

aa = np.transpose(aa, (3, 0, 1, 2))
xf=aa
#xf = np.concatenate([aa, aa], axis=0)
xf = np.transpose(xf, (1, 0, 2, 3))
xf = torch.FloatTensor(xf)
#xf = torch.FloatTensor(xf).unsqueeze(0)

out =  Conv2d(15, 32, kernel_size=(7, 7), stride=1, padding=3)(xf)