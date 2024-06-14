import torch
import torch.nn as nn


class SABlock(nn.Module):
    def __init__(self, channel):
        super(SABlock, self).__init__()
        self.inter_channel = channel // 2
        self.conv_phi = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,
                                  padding=0, bias=False)
        self.conv_theta = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,
                                    padding=0, bias=False)
        self.conv_g = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,
                                padding=0, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.conv_mask = nn.Conv2d(in_channels=self.inter_channel, out_channels=channel, kernel_size=1, stride=1,
                                   padding=0, bias=False)

    def forward(self, x):
        # [N, C, H , W]
        b, c, h, w = x.size()
        # [N, C/2, H * W]
        x_phi = self.conv_phi(x).view(b, self.inter_channel, -1)
        # [N, H * W, C/2]
        x_theta = self.conv_theta(x).view(b, self.inter_channel, -1).permute(0, 2, 1).contiguous()
        x_g = self.conv_g(x).view(b, self.inter_channel, -1).permute(0, 2, 1).contiguous()
        # [N, H * W, H * W]
        f = torch.matmul(x_theta, x_phi)
        # dim = -1 按行进行softmax
        f_div_C = self.softmax(f)

        # [N, H * W, C/2]
        y = torch.matmul(f_div_C, x_g)
        y = y.permute(0, 2, 1).contiguous()
        # [N, C/2, H, W]
        y = y.view(b, self.inter_channel, h, w)
        # [N, C, H , W]
        mask = self.conv_mask(y)
        return mask


if __name__ == '__main__':
    model = SABlock(256).cuda()
    x = torch.randn((1, 256, 256, 256)).cuda()
    print(model(x).size())
