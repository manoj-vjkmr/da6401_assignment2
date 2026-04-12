"""Inference script for visualizing predictions."""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

from models.multitask import MultiTaskPerceptionModel

def process_image(image_path, device):
    """Load and preprocess the image."""
    image = np.array(Image.open(image_path).convert("RGB"))
    
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    # Just for visualization (resized to match network input shape but unnormalized)
    vis_transform = A.Compose([A.Resize(224, 224)])
    vis_image = vis_transform(image=image)['image']
    
    transformed = transform(image=image)
    tensor = transformed['image'].unsqueeze(0).to(device)
    
    return tensor, vis_image

def visualize_predictions(image, logits, bbox, mask_logits):
    """Plot the multi-task outputs using strictly matplotlib."""
    # 1. Classification
    pred_class = torch.argmax(logits, dim=1).item()
    
    # 2. Localization [x_center, y_center, width, height]
    bbox = bbox.squeeze(0).cpu().numpy()
    x_center, y_center, width, height = bbox
    
    # Calculate bottom-left corner for matplotlib patches
    xmin = x_center - (width / 2)
    ymin = y_center - (height / 2)

    # 3. Segmentation
    pred_mask = torch.argmax(mask_logits, dim=1).squeeze(0).cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Left subplot: Image + Bounding Box
    axes[0].imshow(image)
    rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='red', facecolor='none')
    axes[0].add_patch(rect)
    axes[0].text(xmin, ymin - 10, f"Class ID: {pred_class}", color='red', fontsize=12, fontweight='bold', backgroundcolor='white')
    axes[0].set_title("Detection & Classification")
    axes[0].axis("off")
    
    # Right subplot: Segmentation Mask
    axes[1].imshow(pred_mask, cmap="viridis")
    axes[1].set_title("Segmentation Mask")
    axes[1].axis("off")
    
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--weights", type=str, required=True, help="Path to multitask_final.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = MultiTaskPerceptionModel().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    tensor, vis_image = process_image(args.image, device)

    with torch.no_grad():
        outputs = model(tensor)
    
    visualize_predictions(vis_image, outputs['classification'], outputs['localization'], outputs['segmentation'])

if __name__ == "__main__":
    main()