import os
import argparse
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import wandb
from PIL import Image
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from data.pets_dataset import OxfordIIITPetDataset
from models.multitask import MultiTaskPerceptionModel
from models.classification import VGG11Classifier

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, choices=['feature_maps', 'detect', 'segment', 'wild'])
    parser.add_argument('--weights', type=str, default='checkpoints/multitask.pth')
    parser.add_argument('--wild_dir', type=str, default='./custom_pets/', help='Folder of downloaded internet images')
    return parser.parse_args()

def load_image(image_path):

    image = np.array(Image.open(image_path).convert("RGB"))
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    tensor = transform(image=image)['image'].unsqueeze(0)
    return tensor, image

def compute_iou(boxA, boxB):

    xA = max(boxA[0] - boxA[2]/2, boxB[0] - boxB[2]/2)
    yA = max(boxA[1] - boxA[3]/2, boxB[1] - boxB[3]/2)
    xB = min(boxA[0] + boxA[2]/2, boxB[0] + boxB[2]/2)
    yB = min(boxA[1] + boxA[3]/2, boxB[1] + boxB[3]/2)
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

def compute_dice(pred, target):

    pred_flat = pred.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)
    intersection = (pred_flat == target_flat).sum().item()
    return (2. * intersection) / (len(pred_flat) + len(target_flat) + 1e-6)

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    wandb.init(project="da6401-assignment-2", name=f"eval_{args.task}")

    if args.task == 'feature_maps':
        print("Running Feature Map Extraction...")
        model = VGG11Classifier().to(device)
        model.load_state_dict(torch.load("checkpoints/classifier.pth", map_location=device, weights_only=False))
        model.eval()

        dataset = OxfordIIITPetDataset(root_dir='./data/oxford-iiit-pet', split="trainval")
        image_tensor, _, _, _ = dataset[0]
        
        with torch.no_grad():
            _, features = model.encoder(image_tensor.unsqueeze(0).to(device), return_features=True)
            
        fig, axes = plt.subplots(2, 8, figsize=(15, 4))
        plt.suptitle("Task 2.4: Low-level (relu1) vs High-level (relu5) Features")
        
        for i in range(8):
            ax = axes[0, i]
            f_map = features['relu1'][0, i].cpu().numpy()
            ax.imshow(f_map, cmap='viridis')
            ax.axis('off')
            if i == 0: ax.set_title("Relu1 (Edges/Colors)")
            
        for i in range(8):
            ax = axes[1, i]
            f_map = features['relu5'][0, i].cpu().numpy()
            ax.imshow(f_map, cmap='viridis')
            ax.axis('off')
            if i == 0: ax.set_title("Relu5 (Semantic/Shapes)")

        plt.savefig("feature_maps.png")
        wandb.log({"2.4 Feature Maps": wandb.Image("feature_maps.png")})
        print("Logged Feature Maps to W&B.")

    elif args.task == 'detect':
        print("Running Object Detection Evaluation...")
        model = MultiTaskPerceptionModel().to(device)
        model.load_state_dict(torch.load("checkpoints/multitask.pth", map_location=device, weights_only=False))
        model.eval()

        dataset = OxfordIIITPetDataset(root_dir='./data/oxford-iiit-pet', split="test")
        
        columns = ["Image", "Confidence", "IoU"]
        detection_table = wandb.Table(columns=columns)

        for i in range(10):
            image_t, label, gt_bbox, _ = dataset[i]
            with torch.no_grad():
                out = model(image_t.unsqueeze(0).to(device))
                
                pred_bbox = out['localization'][0].cpu().numpy()
                gt_bbox = gt_bbox.numpy()
                
                probs = F.softmax(out['classification'][0], dim=0)
                conf = torch.max(probs).item()
                
                iou = compute_iou(pred_bbox, gt_bbox)

            fig, ax = plt.subplots(1)
            img_vis = image_t.permute(1, 2, 0).numpy()
            img_vis = np.clip((img_vis * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406], 0, 1)
            ax.imshow(img_vis)

            px, py, pw, ph = pred_bbox
            gx, gy, gw, gh = gt_bbox
            ax.add_patch(patches.Rectangle((px-pw/2, py-ph/2), pw, ph, linewidth=2, edgecolor='r', facecolor='none', label='Pred'))
            ax.add_patch(patches.Rectangle((gx-gw/2, gy-gh/2), gw, gh, linewidth=2, edgecolor='g', facecolor='none', label='GT'))
            ax.legend()
            
            plt.savefig("temp_det.png")
            plt.close()
            
            detection_table.add_data(wandb.Image("temp_det.png"), f"{conf:.2f}", f"{iou:.2f}")
            
        wandb.log({"2.5 Detection Results": detection_table})
        print("Logged Detection Table to W&B.")

    elif args.task == 'segment':
        print("Running Segmentation Evaluation...")
        model = MultiTaskPerceptionModel().to(device)
        model.load_state_dict(torch.load("checkpoints/multitask.pth", map_location=device, weights_only=False))
        model.eval()

        dataset = OxfordIIITPetDataset(root_dir='./data/oxford-iiit-pet', split="test")
        
        columns = ["Image", "Ground Truth", "Prediction", "Pixel Acc", "Dice Score"]
        seg_table = wandb.Table(columns=columns)

        for i in range(5):
            image_t, _, _, gt_mask = dataset[i]
            with torch.no_grad():
                out = model(image_t.unsqueeze(0).to(device))
                pred_mask = torch.argmax(out['segmentation'][0], dim=0).cpu()
                
            gt_mask_np = gt_mask.numpy()
            pred_mask_np = pred_mask.numpy()
            
            pixel_acc = (pred_mask_np == gt_mask_np).mean()
            dice = compute_dice(pred_mask, gt_mask)

            img_vis = image_t.permute(1, 2, 0).numpy()
            img_vis = np.clip((img_vis * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406], 0, 1)
            
            seg_table.add_data(
                wandb.Image(img_vis),
                wandb.Image(gt_mask_np * 127),
                wandb.Image(pred_mask_np * 127),
                f"{pixel_acc:.2%}",
                f"{dice:.2%}"
            )
            
        wandb.log({"2.6 Segmentation Results": seg_table})
        print("Logged Segmentation Table to W&B.")

    elif args.task == 'wild':
        print(f"Running Final Pipeline on custom images in {args.wild_dir}...")
        model = MultiTaskPerceptionModel().to(device)
        model.load_state_dict(torch.load("checkpoints/multitask.pth", map_location=device, weights_only=False))
        model.eval()

        if not os.path.exists(args.wild_dir):
            print(f"Error: Directory {args.wild_dir} not found! Create it and add .jpg files.")
            return

        for filename in os.listdir(args.wild_dir):
            if not filename.endswith(('.jpg', '.jpeg', '.png')): continue
            
            image_path = os.path.join(args.wild_dir, filename)
            tensor, orig_img = load_image(image_path)
            
            with torch.no_grad():
                out = model(tensor.to(device))
                pred_class = torch.argmax(out['classification'][0]).item()
                pred_bbox = out['localization'][0].cpu().numpy()
                pred_mask = torch.argmax(out['segmentation'][0], dim=0).cpu().numpy()

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            fig.suptitle(f"Predicted Breed ID: {pred_class}")
            
            ax1.imshow(orig_img)
            px, py, pw, ph = pred_bbox
            h, w, _ = orig_img.shape
            scale_x, scale_y = w/224.0, h/224.0
            ax1.add_patch(patches.Rectangle(((px-pw/2)*scale_x, (py-ph/2)*scale_y), pw*scale_x, ph*scale_y, linewidth=3, edgecolor='r', facecolor='none'))
            ax1.set_title("Detection")
            ax1.axis('off')

            ax2.imshow(orig_img)
            ax2.imshow(pred_mask, alpha=0.5, cmap='jet')
            ax2.set_title("Segmentation")
            ax2.axis('off')

            plt.savefig(f"wild_{filename}")
            wandb.log({f"2.7 Wild Image: {filename}": wandb.Image(f"wild_{filename}")})
            print(f"Logged {filename} to W&B.")

    wandb.finish()

if __name__ == "__main__":
    main()