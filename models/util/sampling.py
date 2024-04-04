import torch
import torch.nn as nn

from torch.nn import Module


class Upsample(Module):
    def __init__(self, in_channels, out_channels, scale_factor, kernel_size=3, batch_norm=True):
        super(Upsample, self).__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            stride=scale_factor, 
            padding=1, 
            output_padding=scale_factor-1
        )
        self.actv = nn.ReLU()
        if batch_norm:
            self.batch_normal = nn.BatchNorm2d(out_channels)


    def forward(self, x):
        x = self.conv(x)
        x = self.actv(x)
        if self.batch_normal:
            x = self.batch_normal(x)
        return x


class Downsample(Module):
    def __init__(self, in_channels, out_channels, scale_factor=2, kernel_size=3, batch_norm=True):
        super(Downsample, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1)
        self.actv = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=scale_factor)
        if batch_norm:
            self.batch_normal = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.actv(x)
        x = self.pool(x)
        if self.batch_normal:
            x = self.batch_normal(x)
        return x