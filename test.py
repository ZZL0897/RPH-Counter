import argparse
import json
import os
import sys
import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader
import exp_configs
from box_dataset import BoxDataset
from utils import models

cudnn.benchmark = True
torch.multiprocessing.set_sharing_strategy('file_system')


def val(exp_dict, datadir, model_path):
    if (sys.platform == 'linux') or (sys.platform == 'linux2'):
        num_workers = 8
    else:
        num_workers = 0
    dir_name = os.path.dirname(model_path)
    print(dir_name)
    # val set
    val_set = BoxDataset(exp_dict['split'], datadir, None)
    val_loader = DataLoader(val_set, batch_size=1, num_workers=num_workers)
    img_names = val_set.img_names

    # Model
    # ==================
    model = models.get_model(exp_dict=exp_dict, train_set=val_set).cuda()
    # if (sys.platform == 'linux') or (sys.platform == 'linux2'):
    #     model = torch.compile(model, mode="reduce-overhead")

    model.load_state_dict(torch.load(model_path))

    # Validate and Visualize the model
    val_dict, result = model.val_on_loader(val_loader,
                                           img_names,
                                           savedir=os.path.join(dir_name, 'vis'),
                                           n_images=100)
    print(val_dict)

    res_name = os.path.join(dir_name, 'res.json')
    with open(res_name, 'w') as json_file:
        json.dump(result, json_file, indent=2)

    print(f"标签数据已保存至 {res_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--exp_group_list', default='box_data', nargs="+")
    parser.add_argument('-m', '--model', default='fcn_resnet_safpn_s16')
    parser.add_argument('-sb', '--savedir_base', default='checkpoints')
    # 如果是测试全部验证集就填数据集根目录，否则填不同密度验证集的根目录
    parser.add_argument('-d', '--datadir', default=r'.\data')
    # 如果是测试全部验证集就填val，如果是测试不同密度验证集就填对应密度验证集的文件夹名称
    parser.add_argument('-sp', '--split', default=r'val')
    parser.add_argument('-mp', '--model_path', default=r'checkpoints\2024-02-21_10-40-08\model_best_f1.pth')

    args = parser.parse_args()
    print(args)

    for exp_group_name in [args.exp_group_list]:
        exp_dict = exp_configs.EXP_GROUPS[exp_group_name][0]
        if args.model is not None:
            exp_dict['model'] = args.model
        if args.split is not None:
            exp_dict['split'] = args.split
        print(exp_dict)

        val(exp_dict=exp_dict,
            datadir=args.datadir,
            model_path=args.model_path)
