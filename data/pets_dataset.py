"""Dataset implementation for Oxford-IIIT Pet."""

import os
import xml.etree.ElementTree as ET
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torch.nn.functional as F

class OxfordIIITPetDataset(Dataset):
    """Oxford-IIIT Pet multi-task dataset loader with fixed resizing."""

    def __init__(self, root_dir: str, split: str = "train", transforms=None):
        """
        Args:
            root_dir: Root directory containing 'images' and 'annotations' folders.
            split: 'train' or 'val'.
            transforms: Albumentations transforms to apply.
        """
        self.root_dir = root_dir
        self.split = split
        self.transforms = transforms

        self.img_dir = os.path.join(root_dir, 'images')
        self.mask_dir = os.path.join(root_dir, 'annotations', 'trimaps')
        self.xml_dir = os.path.join(root_dir, 'annotations', 'xmls')

        # Load trainval.txt
        split_path = os.path.join(root_dir, 'annotations', 'trainval.txt')
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"Missing split file: {split_path}")

        all_samples = []
        self.classes = {}
        
        with open(split_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    img_name = parts[0]
                    class_id = int(parts[1]) - 1  # 0-indexed

                    img_path = os.path.join(self.img_dir, f"{img_name}.jpg")
                    mask_path = os.path.join(self.mask_dir, f"{img_name}.png")
                    xml_path = os.path.join(self.xml_dir, f"{img_name}.xml")

                    if os.path.exists(img_path) and os.path.exists(mask_path) and os.path.exists(xml_path):
                        all_samples.append({
                            'img_name': img_name,
                            'class_id': class_id,
                            'img_path': img_path,
                            'mask_path': mask_path,
                            'xml_path': xml_path
                        })
                        class_name = '_'.join(img_name.split('_')[:-1])
                        if class_id not in self.classes:
                            self.classes[class_id] = class_name

        # 80/20 split
        split_idx = int(len(all_samples) * 0.8)
        if split == 'train':
            self.samples = all_samples[:split_idx]
        else:
            self.samples = all_samples[split_idx:]

    def __len__(self):
        return len(self.samples)

    def _parse_xml(self, xml_path):
        """Parse PASCAL VOC XML."""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = root.find('size')
        width = float(size.find('width').text)
        height = float(size.find('height').text)
        
        bndbox = root.find('object').find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)
        return [xmin, ymin, xmax, ymax], width, height

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 1. Load Data
        image = np.array(Image.open(sample['img_path']).convert("RGB"))
        mask = np.array(Image.open(sample['mask_path'])) - 1  # 0, 1, 2
        bbox, orig_w, orig_h = self._parse_xml(sample['xml_path'])
        
        # 2. Apply Albumentations (Includes Resize to 224, 224)
        if self.transforms:
            transformed = self.transforms(
                image=image, 
                mask=mask, 
                bboxes=[bbox], 
                class_labels=[sample['class_id']]
            )
            image = transformed['image']
            mask = transformed['mask']
            
            if len(transformed['bboxes']) > 0:
                final_bbox = transformed['bboxes'][0]
            else:
                final_bbox = [0, 0, 0, 0]
        else:
            final_bbox = bbox

        # 3. Handle Tensors & Hard Resize Safety
        # If ToTensorV2() was not in transforms, convert manually
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).long()

        # CRITICAL: Final resize check to prevent "storage resize" crash
        if image.shape[1:] != (224, 224):
            image = F.interpolate(image.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)
            # Use nearest for mask to preserve labels
            mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), size=(224, 224), mode='nearest').squeeze().long()

        # 4. Prepare Localization Target (x_center, y_center, width, height)
        # Scaled to 224x224 space
        xmin, ymin, xmax, ymax = final_bbox
        w_box = xmax - xmin
        h_box = ymax - ymin
        x_center = xmin + (w_box / 2.0)
        y_center = ymin + (h_box / 2.0)
        
        target_bbox = torch.tensor([x_center, y_center, w_box, h_box], dtype=torch.float32)

        targets = {
            'classification': torch.tensor(sample['class_id'], dtype=torch.long),
            'localization': target_bbox,
            'segmentation': mask
        }

        return image, targets