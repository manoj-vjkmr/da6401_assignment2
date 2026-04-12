import os
import xml.etree.ElementTree as ET
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

class OxfordIIITPetDataset(Dataset):
    def __init__(self, root_dir: str, split: str = "train", transforms=None):
        self.root_dir = root_dir
        self.split = split
        
        if transforms is not None:
            self.transforms = transforms
        else:
            self.transforms = A.Compose([
                A.Resize(224, 224),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

        self.img_dir = os.path.join(root_dir, 'images')
        self.mask_dir = os.path.join(root_dir, 'annotations', 'trimaps')
        self.xml_dir = os.path.join(root_dir, 'annotations', 'xmls')

        split_path = os.path.join(root_dir, 'annotations', 'trainval.txt')
        all_samples = []
        with open(split_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    img_name = parts[0]
                    img_path = os.path.join(self.img_dir, f"{img_name}.jpg")
                    mask_path = os.path.join(self.mask_dir, f"{img_name}.png")
                    xml_path = os.path.join(self.xml_dir, f"{img_name}.xml")
                    if os.path.exists(img_path) and os.path.exists(mask_path) and os.path.exists(xml_path):
                        all_samples.append({'img_path': img_path, 'mask_path': mask_path, 'xml_path': xml_path, 'class_id': int(parts[1])-1})
        
        split_idx = int(len(all_samples) * 0.8)
        self.samples = all_samples[:split_idx] if split == 'train' else all_samples[split_idx:]

    def _parse_xml(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        bndbox = root.find('object').find('bndbox')
        return [float(bndbox.find(x).text) for x in ['xmin', 'ymin', 'xmax', 'ymax']]

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = np.array(Image.open(sample['img_path']).convert("RGB"))
        mask = np.array(Image.open(sample['mask_path'])) - 1
        bbox = self._parse_xml(sample['xml_path'])

        transformed = self.transforms(
            image=image, 
            mask=mask, 
            bboxes=[bbox], 
            class_labels=[sample['class_id']]
        )
        
        image_tensor = transformed['image']
        mask_tensor = transformed['mask']
        
        if len(transformed['bboxes']) > 0:
            xmin, ymin, xmax, ymax = transformed['bboxes'][0]
            w, h = xmax - xmin, ymax - ymin
            target_bbox = torch.tensor([xmin + w/2, ymin + h/2, w, h], dtype=torch.float32)
        else:
            target_bbox = torch.zeros(4)

        if image_tensor.shape != (3, 224, 224):
             import torch.nn.functional as F
             image_tensor = F.interpolate(image_tensor.unsqueeze(0), size=(224, 224)).squeeze(0)

        return image_tensor, {
            'classification': torch.tensor(sample['class_id'], dtype=torch.long),
            'localization': target_bbox,
            'segmentation': mask_tensor.long()
        }

    def __len__(self):
        return len(self.samples)