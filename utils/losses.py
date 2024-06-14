import random
import torch
import skimage
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from skimage import morphology as morph
from matplotlib import cm as CM
from PIL import Image
from time import time

"""
The Object Counting Loss will be open source after the paper is accepted.
"""


def compute_total_loss(points, probs, boundaries):
    pass


@torch.no_grad()
def get_image_tgt(pr_flat, u_list_per_gt):
    pass


@torch.no_grad()
def get_point_tgt(pt_flat, u_list_per_gt):
    pass


@torch.no_grad()
def get_boundary_tgt(boundaries, points, fg_uniques, blobs, ind_batch):
    pass


def get_fp_tgt(bg_uniques, blobs, ind_batch):
    pass


def build_tgt_dict(scale, ind_batch, ind_list, label):
    return {'scale': scale, 'ind_batch': ind_batch, 'ind_list': ind_list, 'label': label}


class BinaryFocalLoss(nn.Module):
    pass


def bce_loss(pr_flat, tgt_list, loss_func):
    pass


def tensor2float(number):
    if type(number) is torch.Tensor:
        return number.item()
    else:
        return number


def get_blobs(probs, roi_mask=None):
    # 对每个单独区域分配label，从0开始，直接修改矩阵中的区域的数值进行分配
    probs = probs.squeeze()

    pred_mask = (probs > 0.5).astype(int)

    blobs = morph.label(pred_mask == 1)

    if roi_mask is not None:
        blobs = (blobs * roi_mask[None]).astype(int)

    return blobs


def blobs2points(blobs):
    blobs = blobs.squeeze()
    points = np.zeros(blobs.shape).astype("uint8")
    rps = skimage.measure.regionprops(blobs)

    assert points.ndim == 2

    for r in rps:
        y, x = r.centroid

        points[int(y), int(x)] = 1

    return points


def blobs2points_list(blobs):
    blobs = blobs.squeeze()
    points_list = []
    rps = skimage.measure.regionprops(blobs)
    for r in rps:
        y, x = r.centroid
        points_list.append([int(x), int(y)])
    return points_list


def compute_game(pred_points, gt_points, L=1):
    n_rows = 2 ** L
    n_cols = 2 ** L

    pred_points = pred_points.astype(float).squeeze()
    gt_points = np.array(gt_points).astype(float).squeeze()
    h, w = pred_points.shape
    se = 0.

    hs, ws = h // n_rows, w // n_cols
    for i in range(n_rows):
        for j in range(n_cols):
            sr, er = hs * i, hs * (i + 1)
            sc, ec = ws * j, ws * (j + 1)

            pred_count = pred_points[sr:er, sc:ec]
            gt_count = gt_points[sr:er, sc:ec]

            se += float(abs(gt_count.sum() - pred_count.sum()))
    return se


def get_points_from_mask(mask, bg_points=0):
    n_points = 0
    points = np.zeros(mask.shape)
    # print(np.unique(mask))
    assert (len(np.setdiff1d(np.unique(mask), [0, 1, 2])) == 0)

    for c in np.unique(mask):
        if c == 0:
            continue
        blobs = morph.label((mask == c).squeeze())
        points_class = blobs2points(blobs)

        ind = points_class != 0
        n_points += int(points_class[ind].sum())
        points[ind] = c
    assert morph.label((mask).squeeze()).max() == n_points
    points[points == 0] = 255
    if bg_points == -1:
        bg_points = n_points

    if bg_points:
        from haven import haven_utils as hu
        y_list, x_list = np.where(mask == 0)
        with hu.random_seed(1):
            for i in range(bg_points):
                yi = np.random.choice(y_list)
                x_tmp = x_list[y_list == yi]
                xi = np.random.choice(x_tmp)
                points[yi, xi] = 0

    return points
