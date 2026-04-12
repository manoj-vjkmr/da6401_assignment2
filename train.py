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

def calculate_iou(pred_box, true_box):

    x1 = max(pred_box[0], true_box[0])
    y1 = max(pred_box[1], true_box[1])
    x2 = min(pred_box[0] + pred_box[2], true_box[0] + true_box[2])
    y2 = min(pred_box[1] + pred_box[3], true_box[1] + true_box[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = pred_box[2] * pred_box[3]
    box2_area = true_box[2] * true_box[3]
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

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
    parser.add_argument('--epochs', type=int, default=20, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--dropout_p', type=float, default=0.5, help="Dropout probability (Task 2.2)")
    parser.add_argument('--use_batchnorm', action='store_true', help="Enable BatchNorm (Task 2.1)")
    parser.add_argument('--freeze_strategy', type=str, default='full', choices=['full', 'partial', 'strict'], help="Transfer learning strategy (Task 2.3)")
    parser.add_argument('--run_name', type=str, default="Multitask_Baseline", help="Name for W&B Run")
    args = parser.parse_args()

    wandb.init(project="da6401-assignment2", name=args.run_name, config=vars(args))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    DATA_DIR = "./dataset"
    print("Loading datasets...")
    train_dataset = OxfordIIITPetDataset(DATA_DIR, split="train")
    val_dataset = OxfordIIITPetDataset(DATA_DIR, split="val")
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = MultiTaskPerceptionModel().to(device)

    print(f"Applying Freeze Strategy: {args.freeze_strategy}")
    if args.freeze_strategy == 'strict':
        for param in model.encoder.parameters():
            param.requires_grad = False
    elif args.freeze_strategy == 'partial':
        for name, param in model.encoder.named_parameters():
            if "features.16" not in name and "features.18" not in name:
                param.requires_grad = False

    criterion_cls = nn.CrossEntropyLoss()
    criterion_loc = nn.MSELoss()
    criterion_seg = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for images, targets in loop:
            images = images.to(device)
            target_cls = targets['classification'].to(device)
            target_loc = targets['localization'].to(device)
            target_seg = targets['segmentation'].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            
            loss_cls = criterion_cls(outputs['classification'], targets['classification'])
            loss_loc = criterion_loc(outputs['localization'], targets['localization'])
            loss_seg = criterion_seg(outputs['segmentation'], targets['segmentation'])

            total_loss = loss_cls + (loss_loc * 0.001) + loss_seg
            print(f"Cls Loss: {loss_cls.item():.4f} | Loc Loss: {loss_loc.item():.4f} | Seg Loss: {loss_seg.item():.4f}")
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        all_preds_cls, all_true_cls = [], []
        total_pixel_acc, total_dice = 0.0, 0.0
        
        wandb_visuals = []

        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(val_loader):
                images = images.to(device)
                target_cls = targets['classification'].to(device)
                target_loc = targets['localization'].to(device)
                target_seg = targets['segmentation'].to(device)

                outputs = model(images)
                
                loss_cls = criterion_cls(outputs['classification'], target_cls)
                loss_loc = criterion_loc(outputs['localization'], target_loc)
                loss_seg = criterion_seg(outputs['segmentation'], target_seg)
                val_loss += (loss_cls + (loss_loc * 0.1) + loss_seg).item()

                _, preds_cls = torch.max(outputs['classification'], 1)
                all_preds_cls.extend(preds_cls.cpu().numpy())
                all_true_cls.extend(target_cls.cpu().numpy())

                preds_seg = torch.argmax(outputs['segmentation'], dim=1)
                batch_pixel_acc, batch_dice = calculate_segmentation_metrics(preds_seg, target_seg)
                total_pixel_acc += batch_pixel_acc
                total_dice += batch_dice

                if batch_idx == 0:
                    for i in range(min(5, images.size(0))):
                        img_array = images[i].cpu().permute(1, 2, 0).numpy()
                        
                        pred_mask = preds_seg[i].cpu().numpy()
                        true_mask = target_seg[i].cpu().numpy()
                        
                        pred_box = outputs['localization'][i].cpu().numpy()
                        true_box = target_loc[i].cpu().numpy()
                        iou = calculate_iou(pred_box, true_box)

                        w_img = wandb.Image(img_array, 
                            caption=f"IoU: {iou:.2f}",
                            masks={
                                "predictions": {"mask_data": pred_mask, "class_labels": {0: "Background", 1: "Foreground", 2: "Border"}},
                                "ground_truth": {"mask_data": true_mask, "class_labels": {0: "Background", 1: "Foreground", 2: "Border"}}
                            },
                            boxes={
                                "predictions": {"box_data": [{"position": {"minX": float(pred_box[0]), "minY": float(pred_box[1]), "maxX": float(pred_box[0]+pred_box[2]), "maxY": float(pred_box[1]+pred_box[3])}, "class_id": 1, "domain": "pixel"}], "class_labels": {1: "Prediction"}},
                                "ground_truth": {"box_data": [{"position": {"minX": float(true_box[0]), "minY": float(true_box[1]), "maxX": float(true_box[0]+true_box[2]), "maxY": float(true_box[1]+true_box[3])}, "class_id": 2, "domain": "pixel"}], "class_labels": {2: "Ground Truth"}}
                            }
                        )
                        wandb_visuals.append(w_img)

        avg_val_loss = val_loss / len(val_loader)
        val_f1 = f1_score(all_true_cls, all_preds_cls, average='macro')
        avg_pixel_acc = total_pixel_acc / len(val_loader)
        avg_dice = total_dice / len(val_loader)

        print(f"Val Loss: {avg_val_loss:.4f} | Val F1: {val_f1:.4f} | Pixel Acc: {avg_pixel_acc:.4f} | Dice: {avg_dice:.4f}")

        wandb.log({
            "Train Loss": avg_train_loss,
            "Validation Loss": avg_val_loss,
            "Validation F1": val_f1,
            "Pixel Accuracy": avg_pixel_acc,
            "Dice Score": avg_dice,
            "Predictions": wandb_visuals
        })

    print("\nTraining Complete! Splitting weights into required assignment files...")
    model = model.cpu()

    classifier, localizer, unet = VGG11Classifier(), VGG11Localizer(), VGG11UNet()

    classifier.encoder.load_state_dict(model.encoder.state_dict())
    classifier.classifier.load_state_dict(model.classifier_head.state_dict())
    torch.save(classifier.state_dict(), "classifier.pth")

    localizer.encoder.load_state_dict(model.encoder.state_dict())
    localizer.regressor.load_state_dict(model.localizer_head.state_dict())
    torch.save(localizer.state_dict(), "localizer.pth")

    unet.encoder.load_state_dict(model.encoder.state_dict())
    unet.upconv5.load_state_dict(model.seg_upconv5.state_dict())
    unet.dec_block5.load_state_dict(model.seg_dec_block5.state_dict())
    unet.upconv4.load_state_dict(model.seg_upconv4.state_dict())
    unet.dec_block4.load_state_dict(model.seg_dec_block4.state_dict())
    unet.upconv3.load_state_dict(model.seg_upconv3.state_dict())
    unet.dec_block3.load_state_dict(model.seg_dec_block3.state_dict())
    unet.upconv2.load_state_dict(model.seg_upconv2.state_dict())
    unet.dec_block2.load_state_dict(model.seg_dec_block2.state_dict())
    unet.upconv1.load_state_dict(model.seg_upconv1.state_dict())
    unet.dec_block1.load_state_dict(model.seg_dec_block1.state_dict())
    unet.final_conv.load_state_dict(model.seg_final_conv.state_dict())
    torch.save(unet.state_dict(), "unet.pth")

    drive_path = "/content/drive/MyDrive"
    if os.path.exists(drive_path):
        shutil.copy("classifier.pth", os.path.join(drive_path, "classifier.pth"))
        shutil.copy("localizer.pth", os.path.join(drive_path, "localizer.pth"))
        shutil.copy("unet.pth", os.path.join(drive_path, "unet.pth"))
        print("backup uploaded")
    else:
        print("couldnt upload backup")

    wandb.finish()

if __name__ == '__main__':
    main()