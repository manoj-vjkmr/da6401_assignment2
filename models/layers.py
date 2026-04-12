"""Reusable custom layers"""

import torch
import torch.nn as nn

class CustomDropout(nn.Module):
    """Custom Dropout layer using inverted dropout."""

    def __init__(self, p: float = 0.5):
        """
        Initialize the CustomDropout layer.
        Args:
            p: Dropout probability.
        """
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError("Dropout probability must be between 0 and 1.")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the CustomDropout layer.
        Args:
            x: Input tensor for shape [B, C, H, W] or [B, N].
        Returns:
            Output tensor.
        """
        # If model is in evaluation mode or probability is 0, return input as-is
        if not self.training or self.p == 0.0:
            return x
        
        # Create binary mask (1s with probability 1-p, 0s with probability p)
        mask = (torch.rand_like(x) > self.p).float()
        
        # Apply mask and scale using inverted dropout
        return (x * mask) / (1.0 - self.p)