import os
import xml.etree.ElementTree as ET
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

class OxfordIIITPetDataset(Dataset):


    def __init__(self, root_dir: str, split: str = "trainval", transform=None):
        self.root_dir = root_dir
        self.split = split
        
        self.images_dir = os.path.join(root_dir, "images")
        self.annotations_dir = os.path.join(root_dir, "annotations")
        self.trimaps_dir = os.path.join(self.annotations_dir, "trimaps")
        self.xmls_dir = os.path.join(self.annotations_dir, "xmls")
        
        split_file = os.path.join(self.annotations_dir, f"{split}.txt")
        self.samples = []
        
        if not os.path.exists(split_file):
            split_file = os.path.join(root_dir, f"{split}.txt")

        with open(split_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                file_name = parts[0]
                breed_id = int(parts[1]) - 1 
                
                xml_path = os.path.join(self.xmls_dir, f"{file_name}.xml")
                if os.path.exists(xml_path):
                    self.samples.append((file_name, breed_id))

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
        
        img_path = os.path.join(self.images_dir, f"{file_name}.jpg")
        image = np.array(Image.open(img_path).convert("RGB"))
        
        trimap_path = os.path.join(self.trimaps_dir, f"{file_name}.png")
        mask = np.array(Image.open(trimap_path))
        mask = mask - 1 
        
        xml_path = os.path.join(self.xmls_dir, f"{file_name}.xml")
        tree = ET.parse(xml_path)
        root = tree.getroot()
        bndbox = root.find('object').find('bndbox')
        
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)
        
        h_orig, w_orig = image.shape[:2]
        xmin = max(0, min(xmin, w_orig - 1))
        xmax = max(xmin + 1, min(xmax, w_orig))
        ymin = max(0, min(ymin, h_orig - 1))
        ymax = max(ymin + 1, min(ymax, h_orig))

        bboxes = [[xmin, ymin, xmax, ymax]]
        class_labels = [breed_id]
        
        transformed = self.transform(image=image, mask=mask, bboxes=bboxes, class_labels=class_labels)
        image_t = transformed['image']
        mask_t = transformed['mask'].long()
        
        if len(transformed['bboxes']) > 0:
            tx_min, ty_min, tx_max, ty_max = transformed['bboxes'][0]
        else:
            tx_min, ty_min, tx_max, ty_max = 0, 0, 224, 224
            
        w = tx_max - tx_min
        h = ty_max - ty_min
        cx = tx_min + (w / 2.0)
        cy = ty_min + (h / 2.0)
        
        bbox_t = torch.tensor([cx, cy, w, h], dtype=torch.float32)
        
        return image_t, breed_id, bbox_t, mask_t
        