import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.models.base_networks.attention.carafe import CARAFE


class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels, CARF=True):
        super(FPN, self).__init__()

        self.CARF = CARF

        # Lateral Connection
        self.lateral_conv_c2 = nn.Conv2d(in_channels_list[0], out_channels, kernel_size=1)
        self.lateral_conv_c3 = nn.Conv2d(in_channels_list[1], out_channels, kernel_size=1)
        self.lateral_conv_c4 = nn.Conv2d(in_channels_list[2], out_channels, kernel_size=1)
        self.lateral_conv_c5 = nn.Conv2d(in_channels_list[3], out_channels, kernel_size=1)

        if CARF:
            self.carf_p3 = CARAFE(out_channels, out_channels)
            self.carf_p4 = CARAFE(out_channels, out_channels)
            self.carf_p5 = CARAFE(out_channels, out_channels)

    def forward(self, c2, c3, c4, c5):
        # Top-down pathway
        p5 = self.lateral_conv_c5(c5)
        if self.CARF:
            p4 = self.lateral_conv_c4(c4) + self.carf_p5(p5)
            p3 = self.lateral_conv_c3(c3) + self.carf_p4(p4)
            p2 = self.lateral_conv_c2(c2) + self.carf_p3(p3)
        else:
            p4 = self.lateral_conv_c4(c4) + F.interpolate(p5, scale_factor=2, mode='bilinear', align_corners=True)
            p3 = self.lateral_conv_c3(c3) + F.interpolate(p4, scale_factor=2, mode='bilinear', align_corners=True)
            p2 = self.lateral_conv_c3(c2) + F.interpolate(p3, scale_factor=2, mode='bilinear', align_corners=True)

        return p2, p3, p4, p5


if __name__ == '__main__':
    # Example usage:
    # Assuming f2, f3, f4, f5 are your input feature maps
    f2, f3, f4, f5 = torch.rand(1, 256, 64, 64), torch.rand(1, 256, 32, 32), torch.rand(1, 256, 16, 16), torch.rand(1,
                                                                                                                    256,
                                                                                                                    8,
                                                                                                                    8)

    # Define FPN with the corresponding input channels and output channels
    fpn = FPN([256, 256, 256, 256], 256)

    # Forward pass through the FPN
    p2, p3, p4, p5 = fpn(f2, f3, f4, f5)

    # Print the output shapes
    print("Output shape of P2:", p2.shape)
    print("Output shape of P3:", p3.shape)
    print("Output shape of P4:", p4.shape)
    print("Output shape of P5:", p5.shape)
