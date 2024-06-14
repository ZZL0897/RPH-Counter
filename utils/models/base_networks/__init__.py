from . import fcn8_resnet, fcn8_vgg16, fcn8_resnet_s16
from .pspnet.pspnet import PSPNet
from .u2net import U2NET
from .fcn_resnet_fpn import FCNFPN
from .fcn_resnet_fpn_s16 import FCNFPNS16
from .fcn_resnet_pafpn import FCNPaFPN
from .fcn_resnet_pafpn_s16 import FCNPaFPNS16
from .fcn_resnet_safpn import FCNMyFPN
from .fcn_resnet_safpn_s16 import FCNMyFPNS16


def get_base(base_name, exp_dict, n_classes):
    if base_name == "fcn_resnet":
        model = fcn8_resnet.FCN8(n_classes=n_classes)

    elif base_name == "fcn_resnet_s16":
        model = fcn8_resnet_s16.FCN8_s16(n_classes=n_classes)

    elif base_name == "fcn_resnet_fpn":
        model = FCNFPN(n_classes=n_classes)

    elif base_name == "fcn_resnet_fpn_s16":
        model = FCNFPNS16(n_classes=n_classes)

    elif base_name == "fcn_resnet_pafpn":
        model = FCNPaFPN(n_classes=n_classes)

    elif base_name == "fcn_resnet_pafpn_s16":
        model = FCNPaFPNS16(n_classes=n_classes)

    elif base_name == "fcn_resnet_safpn":
        model = FCNMyFPN(n_classes=n_classes)

    elif base_name == "fcn_resnet_safpn_s16":
        model = FCNMyFPNS16(n_classes=n_classes)

    elif base_name == "fcn8_vgg16":
        model = fcn8_vgg16.FCN8_VGG16(n_classes=n_classes)

    elif base_name == 'PSPNet':
        model = PSPNet(layers=50, bins=(1, 2, 3, 6), dropout=0.1, classes=n_classes, zoom_factor=1, use_ppm=True,
                       pretrained=True)

    elif base_name == 'u2net':
        model = U2NET()

    else:
        raise ValueError('%s does not exist' % base_name)

    return model