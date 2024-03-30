import torch
import torch.nn as nn
import torch.optim as optim

from torch import Tensor, tensor
from torch.nn import Module, Parameter

import models.resunet as resunet
from models.ViT import *


@DeprecationWarning
class RightHalfUnet(nn.Module):
    '''
    This model is deprecated, this model structure is considered risky and the implementation is incomplete. Please do not use this model.
    '''
    
    # Channel: 768*2 -> 768 -> 384 -> 192
    # Spatial: 7*7 -> 14*14 -> 28*28 -> 56*56
    
    def __init__(self, patch_size: int, init_ch = 48, num_levels = 5, out_ch = 1):
        super(RightHalfUnet, self).__init__()
        self.patch_size = patch_size
        
        self.decoder = nn.ModuleList()
        for i in range(num_levels, 0, -1):
            self.decoder.append(resunet._ResNetBlock(2**i * init_ch, 'up'))
        self.decoder.append(resunet._ResNetBlock(init_ch, 'none'))
        
        self.output = resunet._Conv2DLayer(init_ch, out_ch, is_output=True)
    
    def forward(self, x: Tensor) -> Tensor:
        up16x, up8x, up4x, up2x, idt, down2x = x.unbind()
        
        # low level features
        x = self.decoder[0](down2x)
        x = self.decoder[1](x + idt)
        x = self.decoder[2](x + up2x)
        x = self.decoder[3](x + up4x)
        x = self.decoder[4](x + up8x)
        x = self.decoder[5](x + up16x)
        
        x = self.output(x)
        return x