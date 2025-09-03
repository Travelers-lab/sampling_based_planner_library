import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import ConvBlock

class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, base_channels=64):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.enc4 = ConvBlock(base_channels * 4, base_channels * 8)

        self.pool = nn.MaxPool2d(2)
        self.up = lambda x: F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

        self.dec3 = ConvBlock(base_channels * 8 + base_channels * 4, base_channels * 4)
        self.dec2 = ConvBlock(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.dec1 = ConvBlock(base_channels * 2 + base_channels, base_channels)
        self.final = nn.Conv2d(base_channels, out_channels, kernel_size=1)


    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))
        out = self.final(d1)


        out = torch.sigmoid(out)

        return out