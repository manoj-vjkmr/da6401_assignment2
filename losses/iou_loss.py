"""Custom IoU loss"""

import torch
import torch.nn as nn

class IoULoss(nn.Module):
    """IoU loss for bounding box regression."""

    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        """
        Initialize the IoULoss module.
        Args:
            eps: Small value to avoid division by zero.
            reduction: Specifies the reduction to apply to the output: 'mean' | 'sum' | 'none'.
        """
        super().__init__()
        self.eps = eps
        
        if reduction not in ["none", "mean", "sum"]:
            raise ValueError("Reduction must be 'none', 'mean', or 'sum'.")
        self.reduction = reduction

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Compute IoU loss between predicted and target bounding boxes.
        Args:
            pred_boxes: [B, 4] predicted boxes in (x_center, y_center, width, height) format.
            target_boxes: [B, 4] target boxes in (x_center, y_center, width, height) format.
        """
        # Convert (cx, cy, w, h) to (x1, y1, x2, y2)
        pred_x1 = pred_boxes[:, 0] - (pred_boxes[:, 2] / 2)
        pred_y1 = pred_boxes[:, 1] - (pred_boxes[:, 3] / 2)
        pred_x2 = pred_boxes[:, 0] + (pred_boxes[:, 2] / 2)
        pred_y2 = pred_boxes[:, 1] + (pred_boxes[:, 3] / 2)

        tgt_x1 = target_boxes[:, 0] - (target_boxes[:, 2] / 2)
        tgt_y1 = target_boxes[:, 1] - (target_boxes[:, 3] / 2)
        tgt_x2 = target_boxes[:, 0] + (target_boxes[:, 2] / 2)
        tgt_y2 = target_boxes[:, 1] + (target_boxes[:, 3] / 2)

        # Calculate Intersection Coordinates
        inter_x1 = torch.max(pred_x1, tgt_x1)
        inter_y1 = torch.max(pred_y1, tgt_y1)
        inter_x2 = torch.min(pred_x2, tgt_x2)
        inter_y2 = torch.min(pred_y2, tgt_y2)

        # Calculate Intersection Area
        inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
        inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
        intersection = inter_w * inter_h

        # Calculate Union Area
        pred_area = torch.clamp(pred_x2 - pred_x1, min=0) * torch.clamp(pred_y2 - pred_y1, min=0)
        tgt_area = torch.clamp(tgt_x2 - tgt_x1, min=0) * torch.clamp(tgt_y2 - tgt_y1, min=0)
        union = pred_area + tgt_area - intersection

        # Calculate IoU and Loss
        iou = intersection / (union + self.eps)
        loss = 1.0 - iou

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        
        return loss