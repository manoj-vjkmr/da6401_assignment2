"""Localization modules"""

import torch
import torch.nn as nn
from .vgg11 import VGG11Encoder
from .layers import CustomDropout

class VGG11Localizer(nn.Module):
    """VGG11-based localizer."""

    def __init__(self, in_channels: int = 3, dropout_p: float = 0.5):
        """Initialize the VGG11Localizer model."""
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels)
        
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(1024, 4),
            nn.Sigmoid()  # Restricts values between 0 and 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for localization model."""
        bottleneck = self.encoder(x, return_features=False)
        
        # Get normalized coordinates [0, 1]
        out = self.regressor(bottleneck)
        
        # Scale back to original 224x224 pixel space: [x_center, y_center, width, height]
        return out * 224.0