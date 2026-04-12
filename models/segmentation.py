"""Segmentation model"""

import torch
import torch.nn as nn
from .vgg11 import VGG11Encoder

class VGG11UNet(nn.Module):
    """U-Net style segmentation network."""

    def __init__(self, num_classes: int = 3, in_channels: int = 3, dropout_p: float = 0.5):
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels)
        
        def up_block(in_c, skip_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c + skip_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        # Transposed Convolutions & Decoder Blocks
        # Stage 5
        self.upconv5 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec5 = up_block(512, 512, 512) # up(512) + skip(relu5: 512)
        
        # Stage 4
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = up_block(256, 512, 256) # up(256) + skip(relu4: 512)
        
        # Stage 3
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = up_block(128, 256, 128) # up(128) + skip(relu3: 256)
        
        # Stage 2
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = up_block(64, 128, 64)   # up(64) + skip(relu2: 128)
        
        # Stage 1 (Final)
        self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(64 + 64, num_classes, kernel_size=1) # up(64) + skip(relu1: 64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck, features = self.encoder(x, return_features=True)
        
        # Decoder path with skip connections via concatenation
        d5 = self.upconv5(bottleneck)
        d5 = torch.cat((d5, features['relu5']), dim=1)
        d5 = self.dec5(d5)
        
        d4 = self.upconv4(d5)
        d4 = torch.cat((d4, features['relu4']), dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.upconv3(d4)
        d3 = torch.cat((d3, features['relu3']), dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, features['relu2']), dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, features['relu1']), dim=1)
        
        out = self.final_conv(d1)
        return out