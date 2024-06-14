import torch
import torch.nn as nn

"""
卷积块
"""


class Conv_Block(nn.Module):
    def __init__(self, in_c, out_c, dilation=1) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, 1, padding=1 * dilation, dilation=1 * dilation),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)


"""
下采样块
"""


class Down_Sample_Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.MaxPool2d(2, 2)
        )

    def forward(self, x):
        return self.layers(x)


"""
上采样块
"""


class Up_Sample_Block(nn.Module):
    def __init__(self, scale_factor=2) -> None:
        super().__init__()
        # 上采样方法1：
        self.upsample1 = nn.UpsamplingBilinear2d(scale_factor=scale_factor)
        # 上采样方法2：
        self.upsample2 = nn.Upsample(scale_factor=scale_factor, mode='bilinear')

    def forward(self, x, feature):
        # 方法3：x = torch.nn.functional.interpolate(input=x,scale_factor=2, mode="nearest")
        x = self.upsample1(x)
        # 下面两行代码是将
        # resize = Resize((x.shape[2], x.shape[3]))
        # feature = resize(feature)
        res = torch.cat((x, feature), dim=1)
        return res


"""
输出模块：
"""


class Output(nn.Module):
    def __init__(self, in_c, out_c) -> None:
        super().__init__()
        self.layers = self.layers = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


class UNet_Block1(nn.Module):
    def __init__(self, in_c, mid_c, out_c) -> None:
        super().__init__()
        self.down = Down_Sample_Block()
        self.up = Up_Sample_Block()

        self.conv1 = Conv_Block(in_c, out_c)
        self.conv2 = Conv_Block(out_c, mid_c)
        self.conv3 = Conv_Block(mid_c, mid_c)
        self.conv4 = Conv_Block(mid_c, mid_c, dilation=2)
        self.conv5 = Conv_Block(mid_c * 2, mid_c)
        self.conv6 = Conv_Block(mid_c * 2, out_c)

    def forward(self, x):
        # 下采样过程
        out1 = self.conv1(x)  # out_c [1, 5, 224, 224]
        out2 = self.conv2(out1)  # mid_c [1, 3, 224, 224]
        out3 = self.conv3(self.down(out2))  # mid_c [1, 3, 112, 112]
        out4 = self.conv3(self.down(out3))  # mid_c [1, 3, 56, 56]
        out5 = self.conv3(self.down(out4))  # mid_c [1, 3, 28, 28]
        out6 = self.conv3(self.down(out5))  # mid_c [1, 3, 14, 14]
        out7 = self.conv3(self.down(out6))  # mid_c [1, 3, 7, 7]

        out8 = self.conv4(out7)  # mid_c [1, 3, 7, 7]
        out9 = self.conv5(torch.cat((out7, out8), dim=1))  # mid_c [1, 3, 7, 7]

        # 上采样
        out10 = self.conv5(self.up(out9, out6))  # [1, 3, 14, 14]
        out11 = self.conv5(self.up(out10, out5))  # [1, 3, 28, 28]
        out12 = self.conv5(self.up(out11, out4))  # [1, 3, 56, 56]
        out13 = self.conv5(self.up(out12, out3))  # [1, 3, 112, 112]
        out14 = self.conv6(self.up(out13, out2))  # [1, 3, 224, 224]
        out = out14 + out1  # [1, 5, 224, 224]
        return out


class UNet_Block2(nn.Module):
    def __init__(self, in_c, mid_c, out_c) -> None:
        super().__init__()
        self.down = Down_Sample_Block()
        self.up = Up_Sample_Block()

        self.conv1 = Conv_Block(in_c, out_c)
        self.conv2 = Conv_Block(out_c, mid_c)
        self.conv3 = Conv_Block(mid_c, mid_c)
        self.conv4 = Conv_Block(mid_c, mid_c, dilation=2)
        self.conv5 = Conv_Block(mid_c * 2, mid_c)
        self.conv6 = Conv_Block(mid_c * 2, out_c)

    def forward(self, x):
        # 下采样过程
        out1 = self.conv1(x)  # out_c [1, 5, 112, 112]
        out2 = self.conv2(out1)  # mid_c [1, 3, 112, 112]
        out3 = self.conv3(self.down(out2))  # mid_c [1, 3, 56, 56]
        out4 = self.conv3(self.down(out3))  # mid_c [1, 3, 28, 28]
        out5 = self.conv3(self.down(out4))  # mid_c [1, 3, 14, 14]
        out6 = self.conv3(self.down(out5))  # mid_c [1, 3, 7, 7]

        out8 = self.conv4(out6)  # mid_c [1, 3, 7, 7]
        out9 = self.conv5(torch.cat((out6, out8), dim=1))  # mid_c [1, 3, 7, 7]

        # 上采样
        out10 = self.conv5(self.up(out9, out5))  # [1, 3, 14, 14]
        out11 = self.conv5(self.up(out10, out4))  # [1, 3, 28, 28]
        out12 = self.conv5(self.up(out11, out3))  # [1, 3, 56, 56]
        out13 = self.conv6(self.up(out12, out2))  # [1, 3, 112, 112]
        out = out13 + out1  # [1, 5, 112, 112]
        return out


class UNet_Block3(nn.Module):
    def __init__(self, in_c, mid_c, out_c) -> None:
        super().__init__()
        self.down = Down_Sample_Block()
        self.up = Up_Sample_Block()

        self.conv1 = Conv_Block(in_c, out_c)
        self.conv2 = Conv_Block(out_c, mid_c)
        self.conv3 = Conv_Block(mid_c, mid_c)
        self.conv4 = Conv_Block(mid_c, mid_c, dilation=2)
        self.conv5 = Conv_Block(mid_c * 2, mid_c)
        self.conv6 = Conv_Block(mid_c * 2, out_c)

    def forward(self, x):
        # 下采样过程
        out1 = self.conv1(x)  # out_c [1, 5, 56, 56]
        out2 = self.conv2(out1)  # mid_c [1, 3, 56, 56]
        out3 = self.conv3(self.down(out2))  # mid_c [1, 3, 28, 28]
        out4 = self.conv3(self.down(out3))  # mid_c [1, 3, 14, 14]
        out5 = self.conv3(self.down(out4))  # mid_c [1, 3, 7, 7]

        out8 = self.conv4(out5)  # mid_c [1, 3, 7, 7]
        out9 = self.conv5(torch.cat((out5, out8), dim=1))  # mid_c [1, 3, 7, 7]

        # 上采样
        out10 = self.conv5(self.up(out9, out4))  # [1, 3, 14, 14]
        out11 = self.conv5(self.up(out10, out3))  # [1, 3, 28, 28]
        out12 = self.conv6(self.up(out11, out2))  # [1, 3, 56, 56]
        out = out12 + out1  # [1, 5, 56, 56]
        return out


class UNet_Block4(nn.Module):
    def __init__(self, in_c, mid_c, out_c) -> None:
        super().__init__()
        self.down = Down_Sample_Block()
        self.up = Up_Sample_Block()

        self.conv1 = Conv_Block(in_c, out_c)
        self.conv2 = Conv_Block(out_c, mid_c)
        self.conv3 = Conv_Block(mid_c, mid_c)
        self.conv4 = Conv_Block(mid_c, mid_c, dilation=2)
        self.conv5 = Conv_Block(mid_c * 2, mid_c)
        self.conv6 = Conv_Block(mid_c * 2, out_c)

    def forward(self, x):
        # 下采样过程
        out1 = self.conv1(x)  # out_c [1, 5, 28, 28]
        out2 = self.conv2(out1)  # mid_c [1, 3, 28, 28]
        out3 = self.conv3(self.down(out2))  # mid_c [1, 3, 14, 14]
        out4 = self.conv3(self.down(out3))  # mid_c [1, 3, 7, 7]

        out8 = self.conv4(out4)  # mid_c [1, 3, 7, 7]
        out9 = self.conv5(torch.cat((out4, out8), dim=1))  # mid_c [1, 3, 7, 7]

        # 上采样
        out10 = self.conv5(self.up(out9, out3))  # [1, 3, 14, 14]
        out11 = self.conv6(self.up(out10, out2))  # [1, 3, 28, 28]
        out = out11 + out1  # [1, 5, 28, 28]
        return out


class UNet_Block5(nn.Module):
    def __init__(self, in_c, mid_c, out_c) -> None:
        super().__init__()

        self.conv1 = Conv_Block(in_c, out_c)
        self.conv2 = Conv_Block(out_c, mid_c)
        self.conv3 = Conv_Block(mid_c, mid_c, dilation=2)
        self.conv4 = Conv_Block(mid_c, mid_c, dilation=4)
        self.conv5 = Conv_Block(mid_c, mid_c, dilation=8)
        self.conv6 = Conv_Block(mid_c * 2, mid_c, dilation=4)
        self.conv7 = Conv_Block(mid_c * 2, mid_c, dilation=2)
        self.conv8 = Conv_Block(mid_c * 2, out_c)

    def forward(self, x):
        out1 = self.conv1(x)  # out_c [1, 5, 14, 14]
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out4 = self.conv4(out3)

        out5 = self.conv5(out4)

        out6 = self.conv6(torch.cat((out4, out5), dim=1))
        out7 = self.conv7(torch.cat((out3, out6), dim=1))
        out8 = self.conv8(torch.cat((out2, out7), dim=1))
        out = out8 + out1
        return out


class U2NET(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.down = Down_Sample_Block()
        self.up1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up2 = nn.UpsamplingNearest2d(scale_factor=4)
        self.up3 = nn.UpsamplingNearest2d(scale_factor=8)
        self.up4 = nn.UpsamplingNearest2d(scale_factor=16)
        self.up5 = nn.UpsamplingNearest2d(scale_factor=32)

        self.unet1 = UNet_Block1(3, 32, 64)
        self.unet2 = UNet_Block2(64, 32, 128)
        self.unet3 = UNet_Block3(128, 64, 256)
        self.unet4 = UNet_Block4(256, 128, 512)
        self.unet5 = UNet_Block5(512, 256, 512)

        self.unet6 = UNet_Block5(512, 256, 512)

        self.de_unet1 = UNet_Block1(128, 16, 64)
        self.de_unet2 = UNet_Block2(256, 32, 64)
        self.de_unet3 = UNet_Block3(512, 64, 128)
        self.de_unet4 = UNet_Block4(1024, 128, 256)
        self.de_unet5 = UNet_Block5(1024, 256, 512)

        self.out1 = Output(64, 1)
        self.out2 = Output(64, 1)
        self.out3 = Output(128, 1)
        self.out4 = Output(256, 1)
        self.out5 = Output(512, 1)

    def forward(self, x):
        # 下采样，编码
        conv1 = self.unet1(x)
        en1 = self.down(conv1)
        conv2 = self.unet2(en1)
        en2 = self.down(conv2)
        conv3 = self.unet3(en2)
        en3 = self.down(conv3)
        conv4 = self.unet4(en3)
        en4 = self.down(conv4)
        conv5 = self.unet5(en4)
        en5 = self.down(conv5)

        conv6 = self.unet6(en5)

        # 上采样，解码
        de1 = self.up1(conv6)  # [1, 512, 14, 14]
        conv7 = self.de_unet5(torch.cat((conv5, de1), dim=1))  # [1, 512, 14, 14]
        de2 = self.up1(conv7)  # [1, 512, 28, 28]
        conv8 = self.de_unet4(torch.cat((conv4, de2), dim=1))  # [1, 256, 28, 28]
        de3 = self.up1(conv8)  # [1, 256, 56, 56]
        conv9 = self.de_unet3(torch.cat((conv3, de3), dim=1))  # [1, 128, 56, 56]
        de4 = self.up1(conv9)  # [1, 128, 112, 112]
        conv10 = self.de_unet2(torch.cat((conv2, de4), dim=1))  # [1, 64, 112, 112]
        de5 = self.up1(conv10)  # [1, 64, 224, 224]

        # 输出
        out1 = self.up5(self.out5(conv6))  # [1, 1, 224, 224]
        out2 = self.up4(self.out5(conv7))  # [1, 1, 224, 224]
        out3 = self.up3(self.out4(conv8))  # [1, 1, 224, 224]
        out4 = self.up2(self.out3(conv9))  # [1, 1, 224, 224]
        out5 = self.up1(self.out2(conv10))  # [1, 1, 224, 224]
        out6 = self.out1(de5)  # [1, 1, 224, 224]

        out = (out1 + out2 + out3 + out4 + out5 + out6) / 6

        return out


if __name__ == "__main__":
    x = torch.randn((2, 3, 224, 224))
    conv = U2NET()
    y = conv(x)
    print(y.shape)

