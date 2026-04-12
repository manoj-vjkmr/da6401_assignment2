"""Segmentation model"""

import torch
import torch.nn as nn
from .vgg11 import VGG11Encoder

class VGG11UNet(nn.Module):
    """U-Net style segmentation network."""

    def __init__(self, num_classes: int = 3, in_channels: int = 3, dropout_p: float = 0.5):
        """Initialize the VGG11UNet model."""
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels)
        
        # Decoder 5: 7x7 -> 14x14
        self.upconv5 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec_block5 = nn.Sequential(
            nn.Conv2d(512 + 512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Decoder 4: 14x14 -> 28x28
        self.upconv4 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec_block4 = nn.Sequential(
            nn.Conv2d(512 + 512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Decoder 3: 28x28 -> 56x56
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec_block3 = nn.Sequential(
            nn.Conv2d(256 + 256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Decoder 2: 56x56 -> 112x112
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec_block2 = nn.Sequential(
            nn.Conv2d(128 + 128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Decoder 1: 112x112 -> 224x224
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_block1 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for segmentation model."""
        bottleneck, features = self.encoder(x, return_features=True)
        
        # Decode Level 5
        d5 = self.upconv5(bottleneck)
        d5 = torch.cat([d5, features["enc5"]], dim=1)
        d5 = self.dec_block5(d5)
        
        # Decode Level 4
        d4 = self.upconv4(d5)
        d4 = torch.cat([d4, features["enc4"]], dim=1)
        d4 = self.dec_block4(d4)
        
        # Decode Level 3
        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, features["enc3"]], dim=1)
        d3 = self.dec_block3(d3)
        
        # Decode Level 2
        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, features["enc2"]], dim=1)
        d2 = self.dec_block2(d2)
        
        # Decode Level 1
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, features["enc1"]], dim=1)
        d1 = self.dec_block1(d1)
        
        out = self.final_conv(d1)
        return out