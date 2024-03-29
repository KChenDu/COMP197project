import torch
import torch.nn as nn
import numpy as np

from segmentation_models_pytorch import Unet

class SMPMiTUNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, dropout: float = 0.1, activation: str = 'sigmoid', **kwargs):
        super(SMPMiTUNet, self).__init__()
        self.model = Unet(encoder_name='mit_b1',
                            encoder_weights='imagenet',
                            decoder_channels=(224, 112, 56, 28, 14),
                            encoder_depth=5,
                            in_channels=in_channels,
                            decoder_use_batchnorm=True,
                            classes=out_channels,
                            activation=activation,
                            aux_params=None,
                            **kwargs)

    def forward(self, x):
        x = self.model(x)
        return x