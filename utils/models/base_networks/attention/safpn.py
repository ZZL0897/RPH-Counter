import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.models.base_networks.attention.carafe import CARAFE
from utils.models.base_networks.attention.SABlock import SABlock


class DownSample(nn.Module):
    def __init__(self, down_scale):
        super(DownSample, self).__init__()
        self.down_scale = down_scale

    def forward(self, x):
        b, c, h, w = x.size()
        return F.adaptive_max_pool2d(x, (h//self.down_scale, w//self.down_scale))


class SAFPN(nn.Module):
    def __init__(self, in_channels_list, out_channels, CARF=False, s=32):
        super(SAFPN, self).__init__()

        self.CARF = CARF
        self.s = s

        self.down = DownSample(2)
        # self.down_c2 = DownSample(4)

        # self.SSA_c2 = SABlock(in_channels_list[0])
        self.SSA_c3 = SABlock(in_channels_list[1])
        self.SSA_c4 = SABlock(in_channels_list[2])
        self.SSA_c5 = SABlock(in_channels_list[3])

        # self.SSA_c2_conv = nn.Conv2d(in_channels_list[0], out_channels, kernel_size=1)
        self.SSA_c3_conv = nn.Conv2d(in_channels_list[1], out_channels, kernel_size=1)
        self.SSA_c4_conv = nn.Conv2d(in_channels_list[2], out_channels, kernel_size=1)
        self.SSA_c5_conv = nn.Conv2d(in_channels_list[3], out_channels, kernel_size=1)

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

    def forward(self, c2, c3, c4, c5):

        if self.s == 32:
            f5 = self.SSA_c5_conv(self.SSA_c5(c5) + c5) + self.lateral_conv_c5(c5)
        else:
            f5 = self.SSA_c5_conv(F.interpolate(self.SSA_c5(self.down(c5)), size=c5.size()[2:], mode='bilinear',
                                                align_corners=True) + c5) + self.lateral_conv_c5(c5)

        # f4 = self.SSA_c4_conv(self.SSA_c4(c4) + c4) + self.lateral_conv_c4(c4)
        f4 = self.SSA_c4_conv(F.interpolate(self.SSA_c4(self.down(c4)), size=c4.size()[2:], mode='bilinear',
                                            align_corners=True) + c4) + self.lateral_conv_c4(c4)

        f3 = self.SSA_c3_conv(F.interpolate(self.SSA_c3(self.down(c3)), size=c3.size()[2:], mode='bilinear',
                                            align_corners=True) + c3) + self.lateral_conv_c3(c3)
        # f3 = self.lateral_conv_c3(c3)
        # f2 = self.SSA_c2_conv(F.interpolate(self.SSA_c2(self.down(c2)), size=c2.size()[2:], mode='bilinear', align_corners=True) + c2) + self.lateral_conv_c2(c2)
        f2 = self.lateral_conv_c2(c2)

        p5 = f5
        if self.CARF:
            p4 = f4 + self.carf_p5(p5)
            p3 = f3 + self.carf_p4(p4)
            p2 = f2 + self.carf_p3(p3)
        else:
            p4 = f4 + F.interpolate(p5, size=f4.size()[2:], mode='nearest')
            p3 = f3 + F.interpolate(p4, size=f3.size()[2:], mode='nearest')
            p2 = f2 + F.interpolate(p3, size=f2.size()[2:], mode='nearest')

        return p2, p3, p4, p5
        # return p2*CA_p2, p3*CA_p3, p4*CA_p4, p5*CA_p5, n3*CA_n3, n4*CA_n4, n5*CA_n5


if __name__ == '__main__':
    # Example usage:
    # Assuming f2, f3, f4, f5 are your input feature maps
    f2, f3, f4, f5 = torch.ones(1, 256, 288, 192), \
                     torch.ones(1, 512, 144, 96), \
                     torch.ones(1, 1024, 72, 48), \
                     torch.ones(1, 2048, 72, 48)

    # Define FPN with the corresponding input channels and output channels
    fpn = SAFPN([256, 512, 1024, 2048], 256, s=16)

    # Forward pass through the FPN
    p2, p3, p4, p5 = fpn(f2, f3, f4, f5)

    # Print the output shapes
    print("Output shape of P2:", p2.shape)
    print("Output shape of P3:", p3.shape)
    print("Output shape of P4:", p4.shape)
    print("Output shape of P5:", p5.shape)
