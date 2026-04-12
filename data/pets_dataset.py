import os
import xml.etree.ElementTree as ET
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

class OxfordIIITPetDataset(Dataset):
    def __init__(self, root_dir, split="train", transforms=None):
        self.root_dir = root_dir
        self.split = split
        self.transforms = transforms if transforms else A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

        split_path = os.path.join(root_dir, 'annotations', 'trainval.txt')
        all_samples = []
        with open(split_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    img_name = parts[0]
                    img_path = os.path.join(root_dir, 'images', f"{img_name}.jpg")
                    mask_path = os.path.join(root_dir, 'annotations', 'trimaps', f"{img_name}.png")
                    xml_path = os.path.join(root_dir, 'annotations', 'xmls', f"{img_name}.xml")
                    if os.path.exists(img_path) and os.path.exists(mask_path) and os.path.exists(xml_path):
                        all_samples.append({'img_path': img_path, 'mask_path': mask_path, 'xml_path': xml_path, 'class_id': int(parts[1])-1})
        
        idx = int(len(all_samples) * 0.8)
        self.samples = all_samples[:idx] if split == 'train' else all_samples[idx:]

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = np.array(Image.open(s['img_path']).convert("RGB"))
        mask = np.array(Image.open(s['mask_path'])) - 1
        
        tree = ET.parse(s['xml_path'])
        root = tree.getroot()
        box = [float(root.find('object/bndbox/' + x).text) for x in ['xmin', 'ymin', 'xmax', 'ymax']]

        ts = self.transforms(image=img, mask=mask, bboxes=[box], class_labels=[s['class_id']])
        
        # Bbox conversion
        if len(ts['bboxes']) > 0:
            xm, ym, xmx, ymx = ts['bboxes'][0]
            target_bbox = torch.tensor([xm+(xmx-xm)/2, ym+(ymx-ym)/2, xmx-xm, ymx-ym])
        else:
            target_bbox = torch.zeros(4)

        return ts['image'], {
            'classification': torch.tensor(s['class_id'], dtype=torch.long),
            'localization': target_bbox,
            'segmentation': ts['mask'].long()
        }
