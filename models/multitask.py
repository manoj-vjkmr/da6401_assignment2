import os
import torch
import torch.nn as nn
from .vgg11 import VGG11Encoder
from .classification import VGG11Classifier
from .localization import VGG11Localizer
from .segmentation import VGG11UNet

class MultiTaskPerceptionModel(nn.Module):
    def __init__(self, num_breeds: int = 37, seg_classes: int = 3, in_channels: int = 3, 
                 classifier_path: str = "classifier.pth", 
                 localizer_path: str = "localizer.pth", 
                 unet_path: str = "unet.pth"):
        super().__init__()
        
        self.encoder = VGG11Encoder(in_channels=in_channels)
        cls_temp = VGG11Classifier(num_classes=num_breeds, in_channels=in_channels)
        loc_temp = VGG11Localizer(in_channels=in_channels)
        seg_temp = VGG11UNet(num_classes=seg_classes, in_channels=in_channels)

        self.classifier_head = cls_temp.classifier
        self.localizer_head = loc_temp.regressor
        
        self.seg_upconv5, self.seg_dec5 = seg_temp.upconv5, seg_temp.dec5
        self.seg_upconv4, self.seg_dec4 = seg_temp.upconv4, seg_temp.dec4
        self.seg_upconv3, self.seg_dec3 = seg_temp.upconv3, seg_temp.dec3
        self.seg_upconv2, self.seg_dec2 = seg_temp.upconv2, seg_temp.dec2
        self.seg_upconv1 = seg_temp.upconv1
        self.seg_final_conv = seg_temp.final_conv

        if not os.path.exists(classifier_path) or not os.path.exists(localizer_path) or not os.path.exists(unet_path):
            try:
                import gdown
                gdown.download(id="1ErN9X3BrHDFvRTE1nniTRAbRfRIimfPU", output=classifier_path, quiet=False)
                gdown.download(id="1HSqaWzqC2EIx-2N6YHm4LbKnc5Ky4SGv", output=localizer_path, quiet=False)
                gdown.download(id="1JfScQk7-VKErpVM9pTHxWEB0s_c15ZWC", output=unet_path, quiet=False)
            except Exception as e:
                print(f"Download failed: {e}")

        try:
            if os.path.exists(classifier_path):
                state = torch.load(classifier_path, map_location="cpu", weights_only=False)
                cls_temp.load_state_dict(state)
                self.encoder.load_state_dict(cls_temp.encoder.state_dict())
                self.classifier_head.load_state_dict(cls_temp.classifier.state_dict())

            if os.path.exists(localizer_path):
                state = torch.load(localizer_path, map_location="cpu", weights_only=False)
                loc_temp.load_state_dict(state)
                self.localizer_head.load_state_dict(loc_temp.regressor.state_dict())

            if os.path.exists(unet_path):
                state = torch.load(unet_path, map_location="cpu", weights_only=False)
                seg_temp.load_state_dict(state)
                self.seg_upconv5.load_state_dict(seg_temp.upconv5.state_dict())
                self.seg_dec5.load_state_dict(seg_temp.dec5.state_dict())
                self.seg_upconv4.load_state_dict(seg_temp.upconv4.state_dict())
                self.seg_dec4.load_state_dict(seg_temp.dec4.state_dict())
                self.seg_upconv3.load_state_dict(seg_temp.upconv3.state_dict())
                self.seg_dec3.load_state_dict(seg_temp.dec3.state_dict())
                self.seg_upconv2.load_state_dict(seg_temp.upconv2.state_dict())
                self.seg_dec2.load_state_dict(seg_temp.dec2.state_dict())
                self.seg_upconv1.load_state_dict(seg_temp.upconv1.state_dict())
                self.seg_final_conv.load_state_dict(seg_temp.final_conv.state_dict())

        except Exception as e:
            print(f"Weight load skipped or failed: {e}")

    def forward(self, x):
        bottleneck, features = self.encoder(x, return_features=True)
        
        logits = self.classifier_head(bottleneck)
        
        bbox = self.localizer_head(bottleneck) * 224.0 
        
        d5 = self.seg_dec5(torch.cat([self.seg_upconv5(bottleneck), features["relu5"]], dim=1))
        d4 = self.seg_dec4(torch.cat([self.seg_upconv4(d5), features["relu4"]], dim=1))
        d3 = self.seg_dec3(torch.cat([self.seg_upconv3(d4), features["relu3"]], dim=1))
        d2 = self.seg_dec2(torch.cat([self.seg_upconv2(d3), features["relu2"]], dim=1))
        
        d1_upsampled = self.seg_upconv1(d2)
        d1_combined = torch.cat([d1_upsampled, features["relu1"]], dim=1)
        seg_mask = self.seg_final_conv(d1_combined)
        
        return {
            'classification': logits, 
            'localization': bbox, 
            'segmentation': seg_mask
        }