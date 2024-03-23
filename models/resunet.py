import torch
import torch.nn as nn
import torch.nn.functional as F

class _Conv2DLayer(nn.Module):
    def __init__(self, in_channels, out_channels, is_output=False):
        super(_Conv2DLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=is_output)
        self.is_output = is_output
        if not is_output:
            self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.Sigmoid() if is_output else nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        if not self.is_output:
            x = self.bn(x)
        return x

class _ResNetBlock(nn.Module):
    def __init__(self, out_channels, block_type, bn=True):
        super(_ResNetBlock, self).__init__()
        self.block_type = block_type
        self.conv1 = _Conv2DLayer(out_channels, out_channels)
        self.conv2 = _Conv2DLayer(out_channels, out_channels) 

        # Down-sampling
        if block_type == 'down':
            self.downsample = nn.Sequential(
                nn.Conv2d(out_channels, out_channels * 2, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                nn.BatchNorm2d(out_channels * 2) if bn else nn.Identity()
            )
        # Up-sampling
        elif block_type == 'up':
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(out_channels, out_channels // 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                nn.ReLU(),
                nn.BatchNorm2d(out_channels // 2) if bn else nn.Identity()
            )

    def forward(self, x):
        identity = x
        
        x = self.conv1(x)
        x = self.conv2(x)
        # skip connection
        x += identity

        if self.block_type == 'down':
            x = self.downsample(x)
        elif self.block_type == 'up':
            x = self.upsample(x)
        return x

class ResUNet(nn.Module):
    def __init__(self, init_ch=32, num_levels=3, out_ch=1):
        super(ResUNet, self).__init__()
        self.first_layer = _Conv2DLayer(in_channels=3, out_channels=init_ch)
        self.encoder = nn.ModuleList([_ResNetBlock(init_ch * 2**i,'down') for i in range(num_levels)])
        self.encoder.append(_ResNetBlock(init_ch * 2**num_levels, 'none')) #None type
        self.decoder = nn.ModuleList([_ResNetBlock(2**i * init_ch, 'up') for i in range(num_levels, 0, -1)])
        self.decoder.append(_ResNetBlock(init_ch, 'none')) #None Type
        self.out_layer = _Conv2DLayer(init_ch, out_ch, is_output=True)

    def forward(self, x):
        x = self.first_layer(x)
        skips = []
        
        # Encoder
        print("============= Encoder Part =============")
        for down in self.encoder[:-1]:
            print("Before down-sampling: x.shape = ",x.shape)
            x = down(x)
            print("After down-sampling: x.shape = ",x.shape)
            print("\n")
            skips.append(x)
        print("Before down-sampling: x.shape = ",x.shape)
        x = self.encoder[-1](x)
        print("After down-sampling: x.shape = ",x.shape)
        print("\n")

        # Decoder, connect with sysmetric encoder
        print("============= Decoder Part =============")
        for up, skip in zip(self.decoder[:-1], reversed(skips)):
            print("Before up-sampling: x.shape = ",x.shape)
            x = F.interpolate(x, size=skip.shape[2:], mode='nearest') + skip  # Resize x to match the size of skip
            x = up(x)
            print("After up-sampling: x.shape = ",x.shape)
            print("\n")
        x = self.decoder[-1](x)
        print("After up-sampling: x.shape = ",x.shape)

        print("Final shape: ",self.out_layer(x).shape)
        return self.out_layer(x)