import torch
import torch.nn as nn

class IoULoss(nn.Module):
    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred_boxes, target_boxes):
        # Ensure boxes are [cx, cy, w, h]
        p_x1 = pred_boxes[:, 0] - pred_boxes[:, 2]/2
        p_y1 = pred_boxes[:, 1] - pred_boxes[:, 3]/2
        p_x2 = pred_boxes[:, 0] + pred_boxes[:, 2]/2
        p_y2 = pred_boxes[:, 1] + pred_boxes[:, 3]/2

        t_x1 = target_boxes[:, 0] - target_boxes[:, 2]/2
        t_y1 = target_boxes[:, 1] - target_boxes[:, 3]/2
        t_x2 = target_boxes[:, 0] + target_boxes[:, 2]/2
        t_y2 = target_boxes[:, 1] + target_boxes[:, 3]/2

        inter_x1 = torch.max(p_x1, t_x1)
        inter_y1 = torch.max(p_y1, t_y1)
        inter_x2 = torch.min(p_x2, t_x2)
        inter_y2 = torch.min(p_y2, t_y2)

        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        union_area = (pred_boxes[:, 2] * pred_boxes[:, 3]) + (target_boxes[:, 2] * target_boxes[:, 3]) - inter_area + self.eps
        
        iou = inter_area / union_area
        loss = 1.0 - iou # Range [0, 1]

        if self.reduction == "mean": return loss.mean()
        if self.reduction == "sum": return loss.sum()
        return loss