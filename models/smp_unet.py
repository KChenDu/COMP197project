from typing import List, Optional
import torch
import torch.nn as nn
import numpy as np

from segmentation_models_pytorch import Unet
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from segmentation_models_pytorch.base import SegmentationModel, SegmentationHead

from models.mae import MaskedAutoencoderViT


class SMPMiTUNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, dropout: float = 0.1, activation: str = 'sigmoid', **kwargs):
        super(SMPMiTUNet, self).__init__()
        self.model = Unet(encoder_name='mit_b1',
                            encoder_weights='imagenet',
                            decoder_channels=[224, 112, 56, 28, 14],
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

class ViTEncodedUnet(SegmentationModel):
    def __init__(self, 
                encoder_state_dict: any = None,
                in_channels: int = 3,
                out_channels: tuple[int] = (512, 320, 128, 64),
                encoder_depth: int = 4,
                activation = 'sigmoid', 
                decoder_use_batchnorm: bool = True,
                decoder_channels: List[int] = (256, 128, 64, 32, 16),
                classes: int = 1,
                freeze_encoder: bool = False,
                **kwargs):
        super(ViTEncodedUnet, self).__init__()
        self.encoder = MaskedAutoencoderViT(img_size=224,
                                            patch_size=16,
                                            in_chans=in_channels,
                                            out_chans=out_channels,
                                            embed_dim=768,
                                            depth=4,
                                            decoder_depth=2,
                                            decoder_embed_dim=512,
                                            num_heads=12,
                                            mlp_ratio=4)
        if encoder_state_dict is not None:
            self.encoder.load_state_dict(encoder_state_dict)
        else:
            self.encoder.initialize_weights()
            
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            
        self.decoder = UnetDecoder(
            encoder_channels=(3, 0, 64, 128, 320, 512),
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth + 1,
            use_batchnorm=decoder_use_batchnorm,
            center=False,
            attention_type=None
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        self.classification_head = None

        self.name = "unet-vit"
        self.initialize()

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        self.check_input_shape(x)

        features = self.encoder.forward_feature(x)
        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks
