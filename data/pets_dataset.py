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
        
        image = transformed['image']
        mask = transformed['mask']
        
        if len(transformed['bboxes']) > 0:
            final_bbox = transformed['bboxes'][0]
        else:
            final_bbox = [0, 0, 0, 0]

        xmin, ymin, xmax, ymax = final_bbox
        w_box = xmax - xmin
        h_box = ymax - ymin
        cx = xmin + (w_box / 2.0)
        cy = ymin + (h_box / 2.0)
        
        target_bbox = torch.tensor([cx, cy, w_box, h_box], dtype=torch.float32)

        return image, {
            'classification': torch.tensor(sample['class_id'], dtype=torch.long),
            'localization': target_bbox,
            'segmentation': mask.long()
        }