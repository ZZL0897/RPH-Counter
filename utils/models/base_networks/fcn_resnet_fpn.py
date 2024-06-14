import torch.nn as nn
import torchvision
import torch
from torchinfo import summary
from utils.models.base_networks.attention.pafpn import PAFPN


class FCNFPN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

        resnet50_32s = torchvision.models.resnet50(weights=None)

        resnet50_32s.load_state_dict(torch.load('./resnet50-IMAGENET1K_V1.pth'))

        # Create a linear layer -- we don't need logits in this case
        resnet50_32s.fc = nn.Sequential()

        self.resnet50_32s = resnet50_32s

        self.score_32s_p5 = nn.Conv2d(256, self.n_classes, kernel_size=1)
        self.score_16s_p4 = nn.Conv2d(256, self.n_classes, kernel_size=1)
        self.score_8s_p3 = nn.Conv2d(256, self.n_classes, kernel_size=1)
        self.score_4s = nn.Conv2d(256, self.n_classes, kernel_size=1)

        self.fpn = PAFPN([256, 512, 1024, 2048], 256, bottom_up=False)

        # # FREEZE BATCH NORMS
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad = False
                m.bias.requires_grad = False

    def forward(self, x):
        self.resnet50_32s.eval()
        input_spatial_dim = x.size()[2:]

        x = self.resnet50_32s.conv1(x)
        x = self.resnet50_32s.bn1(x)
        x = self.resnet50_32s.relu(x)
        x = self.resnet50_32s.maxpool(x)

        c2 = x = self.resnet50_32s.layer1(x)

        c3 = x = self.resnet50_32s.layer2(x)

        c4 = x = self.resnet50_32s.layer3(x)

        c5 = x = self.resnet50_32s.layer4(x)

        p2, p3, p4, p5 = self.fpn(c2, c3, c4, c5)

        res_32s = self.score_32s_p5(p5)
        res_16s = self.score_16s_p4(p4) + nn.functional.interpolate(res_32s, scale_factor=2, mode='bilinear', align_corners=True)
        res_8s = self.score_8s_p3(p3) + nn.functional.interpolate(res_16s, scale_factor=2, mode='bilinear', align_corners=True)
        res_4s = self.score_4s(p2) + nn.functional.interpolate(res_8s, scale_factor=2, mode='bilinear', align_corners=True)

        res = nn.functional.interpolate(res_4s, size=input_spatial_dim, mode='bilinear', align_corners=True)

        # print(logits_upsampled.size())
        return res


if __name__ == '__main__':
    net = FCNFPN(1).cuda()
    summary(net, (2, 3, 864, 1152))
    from tools.flops_counter import get_model_complexity_info
    flop, param = get_model_complexity_info(net, (3, 864, 1152), as_strings=True, print_per_layer_stat=False)
    print("GFLOPs: {}".format(flop))
    print("Params: {}".format(param))