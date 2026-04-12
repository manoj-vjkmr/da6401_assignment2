

import torch
import torch.nn as nn
from .vgg11 import VGG11
from .layers import CustomDropout

class VGG11Localizer(nn.Module):


    def __init__(self, in_channels: int = 3, dropout_p: float = 0.5):

        super().__init__()
        self.encoder = VGG11(in_channels=in_channels)
        
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(1024, 4),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        bottleneck = self.encoder(x, return_features=False)
        
        out = self.regressor(bottleneck)
        
        return out * 224.0