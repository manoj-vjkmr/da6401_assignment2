

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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

def remove_batchnorm(module):

    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d) or isinstance(child, nn.BatchNorm1d):
            setattr(module, name, nn.Identity())
        else:
            remove_batchnorm(child)

def apply_freeze_strategy(model, strategy):

    if strategy == 'none':
        return
        
    if hasattr(model, 'encoder'):
        if strategy == 'full':
            for param in model.encoder.parameters():
                param.requires_grad = False
            print("=> Applied Full Freeze to Backbone.")
            
        elif strategy == 'partial':
            blocks_to_freeze = [
                model.encoder.block1, model.encoder.block2, 
                model.encoder.block3, model.encoder.block4
            ]
            for block in blocks_to_freeze:
                for param in block.parameters():
                    param.requires_grad = False
            print("=> Applied Partial Freeze to Backbone (Blocks 1-4 frozen).")

def main():
    args = parse_args()
    
    run_name = args.run_name if args.run_name else f"train_{args.task}"
    wandb.init(project=args.project_name, name=run_name, config=vars(args))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading datasets...")
    # 1. Load the FULL dataset that is guaranteed to have XMLs
    full_dataset = OxfordIIITPetDataset(root_dir=args.data_dir, split="trainval")
    
    # 2. Dynamically split it 80% Train / 20% Val
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    # Use a fixed generator seed so your split is consistent across runs
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print(f"Initializing {args.task} model...")
    if args.task == 'classification':
        model = VGG11Classifier(num_classes=37, dropout_p=args.dropout_p).to(device)
        criterion = nn.CrossEntropyLoss()
        save_path = "checkpoints/classifier.pth"
        
    elif args.task == 'localization':
        model = VGG11Localizer(dropout_p=args.dropout_p).to(device)
        mse_loss = nn.MSELoss()
        iou_loss = IoULoss(reduction="mean")
        save_path = "checkpoints/localizer.pth"
        
    elif args.task == 'segmentation':
        model = VGG11UNet(num_classes=3, dropout_p=args.dropout_p).to(device)
        criterion = nn.CrossEntropyLoss()
        save_path = "checkpoints/unet.pth"
        apply_freeze_strategy(model, args.freeze_strategy)
        
    elif args.task == 'multitask':
        model = MultiTaskPerceptionModel().to(device)
        save_path = "checkpoints/multitask.pth"

    if args.no_batchnorm:
        print("=> Dynamically removing BatchNorm layers for ablation study.")
        remove_batchnorm(model)

    os.makedirs("checkpoints", exist_ok=True)
    
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(trainable_params, lr=args.lr)

    print(f"Starting training for {args.epochs} epochs...")
    model.train()
    
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        
        for batch_idx, (images, labels, bboxes, masks) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            bboxes = bboxes.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            
            if args.task == 'classification':
                preds = model(images)
                loss = criterion(preds, labels)
                
            elif args.task == 'localization':
                preds = model(images)
                
                preds_norm = preds / 224.0
                bboxes_norm = bboxes / 224.0
                loss_mse = mse_loss(preds_norm, bboxes_norm) 
                
                loss_iou = iou_loss(preds, bboxes) 
                
                loss = loss_mse + loss_iou
                
            elif args.task == 'segmentation':
                preds = model(images)
                loss = criterion(preds, masks)
                
            elif args.task == 'multitask':
                out = model(images)
                l_cls = nn.CrossEntropyLoss()(out['classification'], labels)
                l_loc = nn.MSELoss()(out['localization'], bboxes) + (IoULoss()(out['localization'], bboxes) * 10.0)
                l_seg = nn.CrossEntropyLoss()(out['segmentation'], masks)
                loss = l_cls + l_loc + l_seg

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}] Batch {batch_idx}/{len(train_loader)} Loss: {loss.item():.4f}")
                wandb.log({"batch_loss": loss.item()})

        avg_loss = epoch_loss / len(train_loader)
        print(f"--- Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f} ---")
        wandb.log({"epoch": epoch + 1, "epoch_loss": avg_loss})

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    wandb.finish()

if __name__ == "__main__":
    main()