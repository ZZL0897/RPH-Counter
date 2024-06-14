import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.models.base_networks.attention.carafe import CARAFE


class PAFPN(nn.Module):
    def __init__(self, in_channels_list, out_channels, CARF=False, bottom_up=True, s=32):
        super(PAFPN, self).__init__()

        self.CARF = CARF
        self.bottom_up = bottom_up

        # Lateral Connection
        self.lateral_conv_c2 = nn.Conv2d(in_channels_list[0], out_channels, kernel_size=1)
        self.lateral_conv_c3 = nn.Conv2d(in_channels_list[1], out_channels, kernel_size=1)
        self.lateral_conv_c4 = nn.Conv2d(in_channels_list[2], out_channels, kernel_size=1)
        self.lateral_conv_c5 = nn.Conv2d(in_channels_list[3], out_channels, kernel_size=1)

        if CARF:
            self.carf_p3 = CARAFE(out_channels, out_channels)
            self.carf_p4 = CARAFE(out_channels, out_channels)
            if s == 32:
                self.carf_p5 = CARAFE(out_channels, out_channels)
            else:
                self.carf_p5 = nn.Identity()

        if bottom_up:
            self.conv_n2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.conv_n3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
            if s == 32:
                self.conv_n4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
            else:
                self.conv_n4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, c2, c3, c4, c5):
        # Top-down pathway
        p5 = self.lateral_conv_c5(c5)
        if self.CARF:
            p4 = self.lateral_conv_c4(c4) + self.carf_p5(p5)
            p3 = self.lateral_conv_c3(c3) + self.carf_p4(p4)
            p2 = self.lateral_conv_c2(c2) + self.carf_p3(p3)
        else:
            p4 = self.lateral_conv_c4(c4) + F.interpolate(p5, size=c4.size()[2:], mode='nearest')
            p3 = self.lateral_conv_c3(c3) + F.interpolate(p4, size=c3.size()[2:], mode='nearest')
            p2 = self.lateral_conv_c2(c2) + F.interpolate(p3, size=c2.size()[2:], mode='nearest')

        if self.bottom_up:
            n2 = p2
            n3 = self.conv_n2(n2) + p3
            n4 = self.conv_n3(n3) + p4
            n5 = self.conv_n4(n4) + p5

            return p2, p3, p4, p5, n3, n4, n5
        else:
            return p2, p3, p4, p5


if __name__ == '__main__':
    # Example usage:
    # Assuming f2, f3, f4, f5 are your input feature maps
    f2, f3, f4, f5 = torch.rand(1, 256, 288, 192), \
                     torch.rand(1, 512, 144, 96), \
                     torch.rand(1, 1024, 72, 48), \
                     torch.rand(1, 2048, 36, 24)

    # Define FPN with the corresponding input channels and output channels
    fpn = PAFPN([256, 512, 1024, 2048], 256)

    # Forward pass through the FPN
    p2, p3, p4, p5, n3, n4, n5 = fpn(f2, f3, f4, f5)

    # Print the output shapes
    print("Output shape of P2:", p2.shape)
    print("Output shape of P3:", p3.shape)
    print("Output shape of P4:", p4.shape)
    print("Output shape of P5:", p5.shape)
    print("Output shape of N3:", n3.shape)
    print("Output shape of N4:", n4.shape)
    print("Output shape of N5:", n5.shape)
