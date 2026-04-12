"""Dataset skeleton for Oxford-IIIT Pet."""

import os
import xml.etree.ElementTree as ET
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

class OxfordIIITPetDataset(Dataset):
    """Oxford-IIIT Pet multi-task dataset loader skeleton."""

    def __init__(self, root_dir: str, split: str = "trainval", transform=None):
        """
        Initialize the dataset.
        Args:
            root_dir: Path to the dataset root (should contain 'images' and 'annotations' folders).
            split: 'trainval' or 'test'.
            transform: Albumentations composition for augmentations.
        """
        self.root_dir = root_dir
        self.split = split
        
        self.images_dir = os.path.join(root_dir, "images")
        self.annotations_dir = os.path.join(root_dir, "annotations")
        self.trimaps_dir = os.path.join(self.annotations_dir, "trimaps")
        self.xmls_dir = os.path.join(self.annotations_dir, "xmls")
        
        # Load the split list
        split_file = os.path.join(self.annotations_dir, f"{split}.txt")
        self.samples = []
        
        with open(split_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                file_name = parts[0]
                # Breed ID is 1-indexed in the file, we convert to 0-indexed (0 to 36)
                breed_id = int(parts[1]) - 1 
                
                # Check if this specific image has an XML bounding box. 
                # Not all images in Oxford-Pets have XMLs. We only keep those that do for multi-tasking.
                xml_path = os.path.join(self.xmls_dir, f"{file_name}.xml")
                if os.path.exists(xml_path):
                    self.samples.append((file_name, breed_id))

        # Default transform: Resize to 224x224 and Normalize
        if transform is None:
            self.transform = A.Compose([
                A.Resize(224, 224),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
        else:
            self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        file_name, breed_id = self.samples[idx]
        
        # 1. Load Image
        img_path = os.path.join(self.images_dir, f"{file_name}.jpg")
        image = np.array(Image.open(img_path).convert("RGB"))
        
        # 2. Load Segmentation Trimap
        # Trimaps have pixel values 1 (foreground), 2 (background), 3 (not classified/border)
        # We subtract 1 to map to 0, 1, 2 for CrossEntropyLoss
        trimap_path = os.path.join(self.trimaps_dir, f"{file_name}.png")
        mask = np.array(Image.open(trimap_path)) - 1
        
        # 3. Load Bounding Box
        xml_path = os.path.join(self.xmls_dir, f"{file_name}.xml")
        tree = ET.parse(xml_path)
        root = tree.getroot()
        bndbox = root.find('object').find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)
        
        bboxes = [[xmin, ymin, xmax, ymax]]
        class_labels = [breed_id] # Dummy label field for albumentations
        
        # 4. Apply Transformations
        transformed = self.transform(image=image, mask=mask, bboxes=bboxes, class_labels=class_labels)
        image_t = transformed['image']
        mask_t = transformed['mask'].long()
        
        # Albumentations output bbox is still [xmin, ymin, xmax, ymax] but scaled to 224x224
        if len(transformed['bboxes']) > 0:
            tx_min, ty_min, tx_max, ty_max = transformed['bboxes'][0]
        else:
            tx_min, ty_min, tx_max, ty_max = 0, 0, 0, 0 # Fallback 
            
        # Convert to [x_center, y_center, width, height] format in 224x224 pixel space
        w = tx_max - tx_min
        h = ty_max - ty_min
        cx = tx_min + (w / 2.0)
        cy = ty_min + (h / 2.0)
        bbox_t = torch.tensor([cx, cy, w, h], dtype=torch.float32)
        
        return image_t, breed_id, bbox_t, mask_t