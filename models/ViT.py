import torch
from torch import nn, tensor, Tensor
from torch.nn import Module, Parameter

import numpy as np

import detectron2.modeling.backbone.vit as vit
from segmentation_models_pytorch import Unet, UnetPlusPlus

class ViTBlock(vit.Block):
    def __init__(self, dim: int, 
                 num_heads: int, 
                 mlp_ratio: int, 
                 qkv_bias: bool, 
                 drop: float,
                 **kwargs):
        super(ViTBlock, self).__init__(dim, 
                                       num_heads, 
                                       mlp_ratio, 
                                       qkv_bias, 
                                       drop,
                                       **kwargs)
        
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x)
    
class UpSampling(nn.Module):
    def __init__(self, in_ch, out_ch, scale_factor=2):
        super(UpSampling, self).__init__()
        self.scale_factor = scale_factor
        
        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        self.conv1x1 = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        
    def forward(self, x):
        x = self.up(x)
        x = self.conv1x1(x)
        return x
    
class DownSampling(nn.Module):
    def __init__(self, in_ch, out_ch, scale_factor=2):
        super(DownSampling, self).__init__()
        self.scale_factor = scale_factor
        self.conv_pooling = nn.MaxPool2d(kernel_size=scale_factor, stride=scale_factor)
        self.conv1x1 = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        
    def forward(self, x):
        x = self.conv_pooling(x)
        x = self.conv1x1(x)
        return x
