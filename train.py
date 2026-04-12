import os
import shutil
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import numpy as np
from tqdm import tqdm

from data.pets_dataset import OxfordIIITPetDataset
from models.multitask import MultiTaskPerceptionModel
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet
from sklearn.metrics import f1_score

class IoULoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred_boxes, target_boxes):
        pred_x1 = pred_boxes[:, 0] - (pred_boxes[:, 2] / 2)
        pred_y1 = pred_boxes[:, 1] - (pred_boxes[:, 3] / 2)
        pred_x2 = pred_boxes[:, 0] + (pred_boxes[:, 2] / 2)
        pred_y2 = pred_boxes[:, 1] + (pred_boxes[:, 3] / 2)

        tgt_x1 = target_boxes[:, 0] - (target_boxes[:, 2] / 2)
        tgt_y1 = target_boxes[:, 1] - (target_boxes[:, 3] / 2)
        tgt_x2 = target_boxes[:, 0] + (target_boxes[:, 2] / 2)
        tgt_y2 = target_boxes[:, 1] + (target_boxes[:, 3] / 2)

        inter_x1 = torch.max(pred_x1, tgt_x1)
        inter_y1 = torch.max(pred_y1, tgt_y1)
        inter_x2 = torch.min(pred_x2, tgt_x2)
        inter_y2 = torch.min(pred_y2, tgt_y2)

        inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
        inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
        intersection = inter_w * inter_h

        pred_area = torch.clamp(pred_x2 - pred_x1, min=0) * torch.clamp(pred_y2 - pred_y1, min=0)
        tgt_area = torch.clamp(tgt_x2 - tgt_x1, min=0) * torch.clamp(tgt_y2 - tgt_y1, min=0)
        union = pred_area + tgt_area - intersection

        iou = intersection / (union + self.eps)
        return (1.0 - iou).mean()

def calculate_iou_metric(pred_box, true_box):

    p = pred_box * 224.0
    t = true_box * 224.0
    
    p_x1, p_y1 = p[0] - p[2]/2, p[1] - p[3]/2
    p_x2, p_y2 = p[0] + p[2]/2, p[1] + p[3]/2
    t_x1, t_y1 = t[0] - t[2]/2, t[1] - t[3]/2
    t_x2, t_y2 = t[0] + t[2]/2, t[1] + t[3]/2

    x1, y1 = max(p_x1, t_x1), max(p_y1, t_y1)
    x2, y2 = min(p_x2, t_x2), min(p_y2, t_y2)
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (p[2] * p[3]) + (t[2] * t[3]) - inter
    return inter / union if union > 0 else 0

def calculate_segmentation_metrics(pred_mask, true_mask):
    correct_pixels = (pred_mask == true_mask).sum().item()
    total_pixels = true_mask.numel()
    pixel_acc = correct_pixels / total_pixels

    pred_fg = (pred_mask == 1)
    true_fg = (true_mask == 1)
    intersection = (pred_fg & true_fg).sum().item()
    dice = (2. * intersection) / (pred_fg.sum().item() + true_fg.sum().item() + 1e-8)
    return pixel_acc, dice

def main():
    parser = argparse.ArgumentParser(description="Train Multi-Task VGG11")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-4, help="Slightly higher LR for faster escape from plateau")
    parser.add_argument('--dropout_p', type=float, default=0.5)
    parser.add_argument('--freeze_strategy', type=str, default='full', choices=['full', 'partial', 'strict'])
    parser.add_argument('--run_name', type=str, default="Multitask_Run")
    args = parser.parse_args()

    wandb.init(project="da6401-assignment2", name=args.run_name, config=vars(args))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    DATA_DIR = "./dataset"
    train_dataset = OxfordIIITPetDataset(DATA_DIR, split="train")
    val_dataset = OxfordIIITPetDataset(DATA_DIR, split="val")
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = MultiTaskPerceptionModel().to(device)

    print(f"Applying Freeze Strategy: {args.freeze_strategy}")
    if args.freeze_strategy == 'strict':
        for param in model.encoder.parameters(): param.requires_grad = False
    elif args.freeze_strategy == 'partial':
        for name, param in model.encoder.named_parameters():
            if "features.16" not in name and "features.18" not in name: param.requires_grad = False

    criterion_cls = nn.CrossEntropyLoss()
    criterion_loc = IoULoss()
    criterion_seg = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        e_cls, e_loc, e_seg = 0.0, 0.0, 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for images, targets in loop:
            images = images.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}
            
            optimizer.zero_grad()
            
            outputs = model(images)
            
            loss_cls = criterion_cls(outputs['classification'], targets['classification'])
            loss_loc = criterion_loc(outputs['localization'], targets['localization'])
            loss_seg = criterion_seg(outputs['segmentation'], targets['segmentation'])

            loss = loss_cls + (loss_loc * 2.0) + loss_seg
            print(f"Batch Cls: {loss_cls.item():.4f} | Loc: {loss_loc.item():.4f} | Seg: {loss_seg.item():.4f}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            e_cls += loss_cls.item()
            e_loc += loss_loc.item()
            e_seg += loss_seg.item()
            
            loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)
        
        print(f"\n[Epoch {epoch+1} Train] Avg Cls: {e_cls/len(train_loader):.4f} | Avg Loc (IoU Loss): {e_loc/len(train_loader):.4f} | Avg Seg: {e_seg/len(train_loader):.4f}")

        model.eval()
        val_loss = 0.0
        all_preds_cls, all_true_cls = [], []
        total_pixel_acc, total_dice, total_iou = 0.0, 0.0, 0.0

        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(val_loader):
                images = images.to(device)
                targets = {k: v.to(device) for k, v in targets.items()}

                outputs = model(images)
                
                v_loss_cls = criterion_cls(outputs['classification'], targets['classification'])
                v_loss_loc = criterion_loc(outputs['localization'], targets['localization'])
                v_loss_seg = criterion_seg(outputs['segmentation'], targets['segmentation'])
                
                val_loss += (v_loss_cls + (v_loss_loc * 2.0) + v_loss_seg).item()

                _, preds_cls = torch.max(outputs['classification'], 1)
                all_preds_cls.extend(preds_cls.cpu().numpy())
                all_true_cls.extend(targets['classification'].cpu().numpy())

                preds_seg = torch.argmax(outputs['segmentation'], dim=1)
                batch_pixel_acc, batch_dice = calculate_segmentation_metrics(preds_seg, targets['segmentation'])
                total_pixel_acc += batch_pixel_acc
                total_dice += batch_dice
                
                pred_box = outputs['localization'][0].cpu().numpy()
                true_box = targets['localization'][0].cpu().numpy()
                total_iou += calculate_iou_metric(pred_box, true_box)

        avg_val_loss = val_loss / len(val_loader)
        val_f1 = f1_score(all_true_cls, all_preds_cls, average='macro')
        avg_pixel_acc = total_pixel_acc / len(val_loader)
        avg_dice = total_dice / len(val_loader)
        avg_iou = total_iou / len(val_loader)

        print(f"[Epoch {epoch+1} Val] Val Loss: {avg_val_loss:.4f} | F1: {val_f1:.4f} | Dice: {avg_dice:.4f} | IoU: {avg_iou:.4f}\n")

        wandb.log({
            "epoch": epoch + 1,
            "train/loss": avg_train_loss,
            "val/loss": avg_val_loss,
            "val/f1": val_f1,
            "val/dice": avg_dice,
            "val/iou": avg_iou
        })

    print("Training Complete! Saving weights...")
    model = model.cpu()
    torch.save(model.state_dict(), "full_model.pth")
    wandb.finish()

if __name__ == '__main__':
    main()