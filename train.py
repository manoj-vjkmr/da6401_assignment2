import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import wandb

from data.pets_dataset import OxfordIIITPetDataset
from losses.iou_loss import IoULoss

from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet
from models.multitask import MultiTaskPerceptionModel

def parse_args():
    parser = argparse.ArgumentParser(description="Train Visual Perception Models")
    parser.add_argument('--data_dir', type=str, default='./dataset', help='Path to dataset')
    parser.add_argument('--task', type=str, required=True, choices=['classification', 'localization', 'segmentation', 'multitask'], help='Which model to train')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--project_name', type=str, default='da6401-assignment-2')
    
    parser.add_argument('--dropout_p', type=float, default=0.5, help='Dropout probability')
    parser.add_argument('--freeze_strategy', type=str, choices=['none', 'partial', 'full'], default='none', help='Backbone freezing strategy for segmentation')
    parser.add_argument('--no_batchnorm', action='store_true', help='Programmatically disable batch norm for ablation studies')
    parser.add_argument('--run_name', type=str, default=None, help='Custom W&B run name')
    
    return parser.parse_args()

def calculate_iou_accuracy(preds, targets, threshold=0.5):
    with torch.no_grad():
        p_cx, p_cy, p_w, p_h = preds.unbind(dim=-1)
        t_cx, t_cy, t_w, t_h = targets.unbind(dim=-1)

        p_x1, p_y1 = p_cx - p_w / 2, p_cy - p_h / 2
        p_x2, p_y2 = p_cx + p_w / 2, p_cy + p_h / 2
        t_x1, t_y1 = t_cx - t_w / 2, t_cy - t_h / 2
        t_x2, t_y2 = t_cx + t_w / 2, t_cy + t_h / 2

        i_x1 = torch.max(p_x1, t_x1)
        i_y1 = torch.max(p_y1, t_y1)
        i_x2 = torch.min(p_x2, t_x2)
        i_y2 = torch.min(p_y2, t_y2)

        inter_w = torch.clamp(i_x2 - i_x1, min=0)
        inter_h = torch.clamp(i_y2 - i_y1, min=0)
        inter_area = inter_w * inter_h

        union_area = (p_w * p_h) + (t_w * t_h) - inter_area + 1e-6
        iou = inter_area / union_area
        return (iou >= threshold).float().mean() * 100.0

def remove_batchnorm(module):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d) or isinstance(child, nn.BatchNorm1d):
            setattr(module, name, nn.Identity())
        else:
            remove_batchnorm(child)

def apply_freeze_strategy(model, strategy):
    if strategy == 'none': return
    if hasattr(model, 'encoder'):
        if strategy == 'full':
            for param in model.encoder.parameters(): param.requires_grad = False
        elif strategy == 'partial':
            for block in [model.encoder.block1, model.encoder.block2, model.encoder.block3, model.encoder.block4]:
                for param in block.parameters(): param.requires_grad = False

def main():
    args = parse_args()
    run_name = args.run_name if args.run_name else f"train_{args.task}"
    wandb.init(project=args.project_name, name=run_name, config=vars(args))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading dataset and creating splits...")
    full_dataset = OxfordIIITPetDataset(root_dir=args.data_dir, split="trainval")
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    seg_weights = torch.tensor([0.2, 3.0, 3.0]).to(device)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_seg = nn.CrossEntropyLoss(weight=seg_weights)
    iou_loss_fn = IoULoss(reduction="mean")
    mse_loss_fn = nn.MSELoss()

    if args.task == 'classification':
        model = VGG11Classifier(num_classes=37, dropout_p=args.dropout_p).to(device)
        save_path = "checkpoints/classifier.pth"
    elif args.task == 'localization':
        model = VGG11Localizer(dropout_p=args.dropout_p).to(device)
        save_path = "checkpoints/localizer.pth"
    elif args.task == 'segmentation':
        model = VGG11UNet(num_classes=3, dropout_p=args.dropout_p).to(device)
        save_path = "checkpoints/unet.pth"
        apply_freeze_strategy(model, args.freeze_strategy)
    elif args.task == 'multitask':
        model = MultiTaskPerceptionModel().to(device)
        save_path = "checkpoints/multitask.pth"

    if args.no_batchnorm: remove_batchnorm(model)
    os.makedirs("checkpoints", exist_ok=True)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for batch_idx, (images, labels, bboxes, masks) in enumerate(train_loader):
            images, labels, bboxes, masks = images.to(device), labels.to(device), bboxes.to(device), masks.to(device)
            optimizer.zero_grad()
            
            if args.task == 'classification':
                loss = criterion_cls(model(images), labels)
            elif args.task == 'localization':
                preds = model(images)
                loss = mse_loss_fn(preds / 224.0, bboxes / 224.0) + (5.0 * iou_loss_fn(preds, bboxes))
            elif args.task == 'segmentation':
                loss = criterion_seg(model(images), masks)
            elif args.task == 'multitask':
                out = model(images)
                l_cls = criterion_cls(out['classification'], labels)
                l_loc = mse_loss_fn(out['localization']/224.0, bboxes/224.0) + (5.0 * iou_loss_fn(out['localization'], bboxes))
                l_seg = criterion_seg(out['segmentation'], masks)
                loss = (0.2 * l_cls) + l_loc + l_seg

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if batch_idx % 10 == 0: wandb.log({"batch_loss": loss.item()})

        model.eval()
        v_loss, v_iou_acc = 0.0, 0.0
        with torch.no_grad():
            for imgs, lbls, box, msk in val_loader:
                imgs, lbls, box, msk = imgs.to(device), lbls.to(device), box.to(device), msk.to(device)
                if args.task == 'classification':
                    v_loss += criterion_cls(model(imgs), lbls).item()
                elif args.task == 'localization':
                    p = model(imgs)
                    v_loss += (mse_loss_fn(p/224.0, box/224.0) + iou_loss_fn(p, box)).item()
                    v_iou_acc += calculate_iou_accuracy(p, box).item()
                elif args.task == 'segmentation':
                    v_loss += criterion_seg(model(imgs), msk).item()
                elif args.task == 'multitask':
                    o = model(imgs)
                    vl = criterion_cls(o['classification'], lbls) + \
                         (mse_loss_fn(o['localization']/224.0, box/224.0) + iou_loss_fn(o['localization'], box)) + \
                         criterion_seg(o['segmentation'], msk)
                    v_loss += vl.item()
                    v_iou_acc += calculate_iou_accuracy(o['localization'], box).item()

        avg_train, avg_val = epoch_loss/len(train_loader), v_loss/len(val_loader)
        avg_acc = v_iou_acc/len(val_loader) if args.task in ['localization', 'multitask'] else 0
        
        print(f"Epoch {epoch+1}: Train={avg_train:.4f}, Val={avg_val:.4f}, IoU_Acc={avg_acc:.1f}%")
        wandb.log({"epoch": epoch+1, "train_loss": avg_train, "val_loss": avg_val, "val_iou_acc": avg_acc})

    torch.save(model.state_dict(), save_path)
    wandb.finish()

if __name__ == "__main__": main()