"""VGG11 encoder"""

from typing import Dict, Tuple, Union
import torch
import torch.nn as nn

class VGG11Encoder(nn.Module):
    """VGG11-style encoder with intermediate feature returns for UNet skip connections."""

    def __init__(self, in_channels: int = 3):
        """Initialize the VGG11Encoder model."""
        super().__init__()
        
        # Block 1: Output [B, 64, H, W]
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2: Output [B, 128, H/2, W/2]
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 3: Output [B, 256, H/4, W/4]
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 4: Output [B, 512, H/8, W/8]
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 5: Output [B, 512, H/16, W/16]
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        # Final pooling to get the 7x7 spatial dimension for VGG heads
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Adaptive pooling ensures that even if the input is not 224x224, 
        # the output to the Linear layers is always 7x7 spatial.
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward pass."""
        
        # Extract features for skip connections
        f1 = self.block1(x)
        p1 = self.pool1(f1)
        
        f2 = self.block2(p1)
        p2 = self.pool2(f2)
        
        f3 = self.block3(p2)
        p3 = self.pool3(f3)
        
        f4 = self.block4(p3)
        p4 = self.pool4(f4)
        
        f5 = self.block5(p4)
        p5 = self.pool5(f5)
        
        # Final bottleneck for Classification and Localization
        bottleneck = self.avgpool(p5)

        if return_features:
            feature_dict = {
                "enc1": f1,  # Used by UNet skip connection 1
                "enc2": f2,  # Used by UNet skip connection 2
                "enc3": f3,  # Used by UNet skip connection 3
                "enc4": f4,  # Used by UNet skip connection 4
                "enc5": f5   # Used by UNet skip connection 5
            }
            return bottleneck, feature_dict
        
        return bottleneck