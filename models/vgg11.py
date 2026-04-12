"""VGG11Encoder encoder"""

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn

class VGG11Encoder(nn.Module):
    """VGG11Encoder-style encoder with optional intermediate feature returns."""

    def __init__(self, in_channels: int = 3):
        """Initialize the VGG11Encoder model."""
        super().__init__()
        
        def conv_block(in_c, out_c):
            # BatchNorm placed after Conv, before ReLU for optimal feature scaling
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        # Standard VGG11 Topology
        self.block1 = conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.block2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.block3 = nn.Sequential(
            conv_block(128, 256),
            conv_block(256, 256)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.block4 = nn.Sequential(
            conv_block(256, 512),
            conv_block(512, 512)
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.block5 = nn.Sequential(
            conv_block(512, 512),
            conv_block(512, 512)
        )
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        
        features = {}
        
        # Block 1 [224 -> 112]
        x1 = self.block1(x)
        features['relu1'] = x1
        x = self.pool1(x1)
        
        # Block 2 [112 -> 56]
        x2 = self.block2(x)
        features['relu2'] = x2
        x = self.pool2(x2)
        
        # Block 3 [56 -> 28]
        x3 = self.block3(x)
        features['relu3'] = x3
        x = self.pool3(x3)
        
        # Block 4 [28 -> 14]
        x4 = self.block4(x)
        features['relu4'] = x4
        x = self.pool4(x4)
        
        # Block 5 [14 -> 7]
        x5 = self.block5(x)
        features['relu5'] = x5
        out = self.pool5(x5)
        
        if return_features:
            return out, features
        return out