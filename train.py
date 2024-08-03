import argparse
import os
import pprint
import time
from datetime import datetime
import pandas as pd
import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader
import exp_configs
from box_dataset import BoxDataset
from exp_configs import save_json
from utils import models
from torch.utils.tensorboard import SummaryWriter
import sys

cudnn.benchmark = True
torch.multiprocessing.set_sharing_strategy('file_system')


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        # self.log = open(filename, "a")
        self.log = open(filename, "a", encoding="utf-8")  # 防止编码错误

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def train(exp_dict, savedir_base, datadir, num_workers=0):
    # ==================
    exp_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    savedir = os.path.join(savedir_base, exp_id)
    log_writer = SummaryWriter(log_dir=os.path.join(savedir, 'log/'))
    sys.stdout = Logger(os.path.join(savedir, 'log.txt'))

    pprint.pprint(exp_dict)

    os.makedirs(savedir, exist_ok=True)
    save_json(os.path.join(savedir, "exp_dict.json"), exp_dict)
    print("Experiment saved in %s" % savedir)

    # Dataset
    # ==================
    # train set
    if (sys.platform == 'linux') or (sys.platform == 'linux2'):
        num_workers = 8
    train_set = BoxDataset('train', datadir, None)
    train_loader = DataLoader(train_set, batch_size=exp_dict["batch_size"], drop_last=True, num_workers=num_workers,
                              shuffle=True)

    # val set
    val_set = BoxDataset('val', datadir, None)
    val_loader = DataLoader(val_set, batch_size=1, num_workers=num_workers)

    # Model
    # ==================
    model = models.get_model(exp_dict=exp_dict, train_set=train_set).cuda()
    # if (sys.platform == 'linux') or (sys.platform == 'linux2'):
    #     model = torch.compile(model, mode="reduce-overhead")

    score_list = []

    best_mse_e = 0
    best_mse_f1 = 0
    for e in range(1, exp_dict['max_epoch'] + 1):
        # Validate only at the start of each cycle
        score_dict = {}

        # Train the model
        time.sleep(0.1)
        train_dict = model.train_on_loader(train_loader)

        # Validate and Visualize the model
        time.sleep(0.1)
        val_dict, result = model.val_on_loader(val_loader)
        time.sleep(0.1)

        # Get new score_dict
        score_dict.update(val_dict)
        score_dict.update(train_dict)
        score_dict["epoch"] = e

        display_dict = {}
        for key, value in score_dict.items():
            if key.endswith('loss'):
                k = 'loss/' + key
                if key.startswith('train'):
                    display_dict[key] = value
            elif key.startswith('val_m'):
                k = 'val/count/' + key
                display_dict[key] = value
            elif key.startswith('val'):
                k = 'val/f1/' + key
                display_dict[key] = value
            else:
                continue
            log_writer.add_scalar(k, value, e)

        # Add to score_list and save checkpoint
        score_list += [display_dict]

        # Report & Save
        score_df = pd.DataFrame(score_list)
        print("\n", score_df.tail(), "\n")

        # torch.save(model.state_dict(), os.path.join(savedir, str(e) + ".pth"))
        print("Checkpoint Saved: %s" % savedir)

        # Save Best Checkpoint
        # print(score_dict.get("val_mse", 0), score_df["val_mse"][:-1].fillna(0))
        if e == 1 or (score_dict.get("val_mse", 0) < score_df["val_mse"][:-1].fillna(0).min()):
            torch.save(model.state_dict(), os.path.join(savedir, "model_best_mse.pth"))
            best_mse_e = e
            print("Saved Best MSE: %s" % savedir)
        if e == 1 or (score_dict.get("val_f1", 0) > score_df["val_f1"][:-1].fillna(0).max()):
            torch.save(model.state_dict(), os.path.join(savedir, "model_best_f1.pth"))
            best_mse_f1 = e
            print("Saved Best F1: %s" % savedir)
        torch.save(model.state_dict(), os.path.join(savedir, "latest.pth"))
        print("Best MSE: {}  At Epoch {}".format(score_df["val_mse"].fillna(0).min(), best_mse_e))
        print("Best F1: {}  At Epoch {}".format(score_df["val_f1"].fillna(0).max(), best_mse_f1))

    log_writer.close()
    print('Experiment completed et epoch %d' % e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--exp_group_list', default='box_data', nargs="+")
    parser.add_argument('-m', '--model', default='fcn_resnet_safpn_s16')
    parser.add_argument('-sb', '--savedir_base', default='checkpoints')
    parser.add_argument('-d', '--datadir', default=r'D:\Users\ZZL\Desktop\test3')

    args = parser.parse_args()
    print(args)

    for exp_group_name in [args.exp_group_list]:
        exp_dict = exp_configs.EXP_GROUPS[exp_group_name][0]
        if args.model is not None:
            exp_dict['model'] = args.model
        print(exp_dict)

        train(exp_dict=exp_dict,
              savedir_base=args.savedir_base,
              datadir=args.datadir)
