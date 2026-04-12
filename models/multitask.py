import os
import torch
import torch.nn as nn
from .vgg11 import VGG11Encoder
from .classification import VGG11Classifier
from .localization import VGG11Localizer
from .segmentation import VGG11UNet

class MultiTaskPerceptionModel(nn.Module):
    def __init__(self, num_breeds: int = 37, seg_classes: int = 3, in_channels: int = 3, 
                 classifier_path: str = "classifier.pth", localizer_path: str = "localizer.pth", unet_path: str = "unet.pth"):
        super().__init__()
        
        # 1. DEFINE ARCHITECTURE FIRST (Ensures attributes always exist)
        self.encoder = VGG11Encoder(in_channels=in_channels)
        cls_temp = VGG11Classifier(num_classes=num_breeds, in_channels=in_channels)
        loc_temp = VGG11Localizer(in_channels=in_channels)
        seg_temp = VGG11UNet(num_classes=seg_classes, in_channels=in_channels)

        self.classifier_head = cls_temp.classifier
        self.localizer_head = loc_temp.regressor
        
        self.seg_upconv5, self.seg_dec_block5 = seg_temp.upconv5, seg_temp.dec_block5
        self.seg_upconv4, self.seg_dec_block4 = seg_temp.upconv4, seg_temp.dec_block4
        self.seg_upconv3, self.seg_dec_block3 = seg_temp.upconv3, seg_temp.dec_block3
        self.seg_upconv2, self.seg_dec_block2 = seg_temp.upconv2, seg_temp.dec_block2
        self.seg_upconv1, self.seg_dec_block1 = seg_temp.upconv1, seg_temp.dec_block1
        self.seg_final_conv = seg_temp.final_conv

        # 2. DOWNLOAD & LOAD WEIGHTS
        if not os.path.exists(classifier_path):
            try:
                import gdown
                gdown.download(id="1YVvPTe2y5m_-m7Ky733kGjmc4OTxtOMw", output=classifier_path, quiet=False)
                gdown.download(id="1wZtrvM6Ru0kBPk_ZX9EDtcOwSW61lyZU", output=localizer_path, quiet=False)
                gdown.download(id="1zZyAKYB4RgQffXgWv2WNndZ1R0B7O5Sn", output=unet_path, quiet=False)
            except: pass

        try:
            if os.path.exists(classifier_path):
                cls_temp.load_state_dict(torch.load(classifier_path, map_location="cpu"))
                self.encoder.load_state_dict(cls_temp.encoder.state_dict())
            if os.path.exists(localizer_path):
                loc_temp.load_state_dict(torch.load(localizer_path, map_location="cpu"))
            if os.path.exists(unet_path):
                seg_temp.load_state_dict(torch.load(unet_path, map_location="cpu"))
        except Exception as e:
            print(f"Weight load skipped: {e}")

    def forward(self, x):
        bottleneck, features = self.encoder(x, return_features=True)
        logits = self.classifier_head(bottleneck)
        bbox = self.localizer_head(bottleneck)
        
        d5 = self.seg_dec_block5(torch.cat([self.seg_upconv5(bottleneck), features["enc5"]], dim=1))
        d4 = self.seg_dec_block4(torch.cat([self.seg_upconv4(d5), features["enc4"]], dim=1))
        d3 = self.seg_dec_block3(torch.cat([self.seg_upconv3(d4), features["enc3"]], dim=1))
        d2 = self.seg_dec_block2(torch.cat([self.seg_upconv2(d3), features["enc2"]], dim=1))
        d1 = self.seg_dec_block1(torch.cat([self.seg_upconv1(d2), features["enc1"]], dim=1))
        
        return {'classification': logits, 'localization': bbox, 'segmentation': self.seg_final_conv(d1)}
