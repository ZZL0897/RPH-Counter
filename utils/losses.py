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


def compute_total_loss(points, probs, boundaries):
    """

    :param boundaries: preload in DataLoader, the sets of all label boxes, bool_array
    :param points: size:(bs, h, w)
    :param probs: size:(bs, c, h, w) 分割网络最后的激活输出
    :return:
    """
    assert (points.max() <= 1)
    bs, c, h, w = probs.size()

    pt_flat = points.view(bs, -1)  # size: (bs, h*w)
    pr_flat = probs.view(bs, -1)  # size: (bs, h*w)
    u_list_per_gt = [torch.unique(_) for _ in pt_flat]  # size: (bs, unique_count)

    img_loss_func = BinaryFocalLoss()
    pt_loss_func = nn.BCELoss()
    bound_loss_func = nn.BCELoss()
    fp_loss_func = BinaryFocalLoss()

    # image loss与point loss实现了对一整个batch的计算
    img_tgt = get_image_tgt(pr_flat, u_list_per_gt)
    point_tgt = get_point_tgt(pt_flat, u_list_per_gt)
    img_loss = bce_loss(pr_flat, img_tgt, img_loss_func)
    point_loss = bce_loss(pr_flat, point_tgt, pt_loss_func)

    # 计算boundary loss与false positive loss
    probs_numpy_batch = probs.detach().view(bs, h, w).cpu().numpy()  # size: (bs, h, w)

    # old
    # points_batch = points.cpu().numpy()  # size: (bs, h, w)
    # new opt speed
    points_batch = points  # size: (bs, h, w)

    # will del
    # blobs_batch = np.zeros((bs, h, w))

    boundary_loss = 0.
    fp_loss = 0.
    # 未实现对整个batch的计算，使用循环遍历
    for i, probs_numpy in enumerate(probs_numpy_batch):
        blobs = get_blobs(probs_numpy, roi_mask=None)

        blobs = torch.from_numpy(blobs).to(points.device)

        # will del
        # blobs_batch[i] = blobs

        # # 保存激活输出，可视化
        # sm1 = CM.ScalarMappable(cmap=CM.jet)
        # gen_dmap = Image.fromarray((sm1.to_rgba(probs_numpy, bytes=True)[:, :, :3]))
        # gen_dmap.save(r'map1.png')

        # get foreground and background blobs
        fg_uniques = torch.unique(blobs * points_batch[i])
        bg_uniques = [x for x in torch.unique(blobs) if x not in fg_uniques]

        boundary_tgt = get_boundary_tgt(boundaries[i], points_batch[i], fg_uniques, blobs, i)
        fp_tgt = get_fp_tgt(bg_uniques, blobs, i)

        boundary_loss += bce_loss(pr_flat, boundary_tgt, bound_loss_func)
        fp_loss += bce_loss(pr_flat, fp_tgt, fp_loss_func)
    # total_loss = img_loss + point_loss + boundary_loss + fp_loss
    # print(tensor2float(img_loss), tensor2float(point_loss), tensor2float(boundary_loss), tensor2float(fp_loss))
    return img_loss, point_loss, boundary_loss, fp_loss


@torch.no_grad()
def get_image_tgt(pr_flat, u_list_per_gt):
    ind_bg = pr_flat.argmin(dim=1)
    ind_fg = pr_flat.argmax(dim=1)
    tgt_list = []
    for i, u_list in enumerate(u_list_per_gt):
        # 至少有一个点被预测为背景，这里取网络激活输出的最小值位置
        if 0 in u_list:
            # tgt_list += [{'scale': 1, 'ind_list': [ind_bg[i]], 'label': 0}]
            tgt_list += [build_tgt_dict(1, i, [ind_bg[i]], 0)]

        # 至少有一个点被预测为前景，这里取网络激活输出的最大值位置
        # 在网络逐渐优化后，该点极大概率就是真正的前景位置
        if 1 in u_list:
            # ind_fg = torch.where(pt_flat == 1)[0]
            # tgt_list += [{'scale': 1, 'ind_list': [ind_fg[i]], 'label': 1}]
            tgt_list += [build_tgt_dict(1, i, [ind_fg[i]], 1)]

    return tgt_list


@torch.no_grad()
def get_point_tgt(pt_flat, u_list_per_gt):
    tgt_list = []
    for i, u_list in enumerate(u_list_per_gt):
        # point level
        # 将所有点标签位置设置为TP，该点应被预测为前景
        if 1 in u_list:
            ind_fg = torch.where(pt_flat[i] == 1)[0]
            # tgt_list += [{'scale': len(ind_fg), 'ind_list': ind_fg, 'label': 1}]
            tgt_list += [build_tgt_dict(len(ind_fg), i, ind_fg, 1)]

    return tgt_list


@torch.no_grad()
def get_boundary_tgt(boundaries, points, fg_uniques, blobs, ind_batch):
    tgt_list = []
    # split level
    # -----------
    n_total = points.sum()

    if n_total > 1:
        # global split
        # 期望网络能够将所有的实例按照边界分割开来
        ind_bg = torch.where(boundaries)[0]
        # print(1, ind_bg)

        # tgt_list += [{'scale': (n_total - 1), 'ind_list': ind_bg, 'label': 0}]
        tgt_list += [build_tgt_dict((n_total - 1), ind_batch, ind_bg, 0)]

        # local split
        # 寻找将多个实例预测成一个大blobs的情况并再进行优化
        for u in fg_uniques:
            if u == 0:
                continue

            ind = blobs == u

            b_points = points * ind
            n_points = b_points.sum()

            if n_points < 2:
                continue

            # ind_bg_local = torch.where(boundaries * ind.ravel())[0]
            # print(ind_bg_local)
            # print(2, ind_bg)
            # print(3, ind)

            # local split损失有可能为0，进而会造成返回的ind_bg为空，造成损失为nan
            if len(ind_bg) != 0:
                # tgt_list += [{'scale': (n_points - 1), 'ind_list': ind_bg, 'label': 0}]
                tgt_list += [build_tgt_dict((n_points - 1), ind_batch, ind_bg, 0)]
    return tgt_list


def get_fp_tgt(bg_uniques, blobs, ind_batch):
    tgt_list = []
    # 选取模型输出预测为前景，但实际为背景的区域
    # fp_loss优化模型，使其不会错误的预测背景为前景
    for u in bg_uniques:
        if u == 0:
            continue
        b_mask = blobs == u
        if b_mask.sum() == 0:
            pass
        else:
            ind_bg = torch.where(b_mask.ravel())[0]
            # tgt_list += [{'scale': 1, 'ind_list': ind_bg, 'label': 0}]
            # if random.choices([True, False], [0.75, 0.25]):
            #     tgt_list += [build_tgt_dict(1, ind_batch, ind_bg, 0)]
            tgt_list += [build_tgt_dict(1, ind_batch, ind_bg, 0)]
    return tgt_list


def build_tgt_dict(scale, ind_batch, ind_list, label):
    return {'scale': scale, 'ind_batch': ind_batch, 'ind_list': ind_list, 'label': label}


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        # pt = torch.exp(-bce_loss)
        # focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        # if self.reduction == 'mean':
        #     return torch.mean(focal_loss)
        # else:
        #     return focal_loss

        p = inputs
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = bce_loss * ((1 - p_t) ** self.gamma)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * loss
        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        else:
            return focal_loss


def bce_loss(pr_flat, tgt_list, loss_func):
    loss = 0.
    for tgt_dict in tgt_list:
        pr_subset = pr_flat[tgt_dict['ind_batch']][tgt_dict['ind_list']]
        # pr_subset = pr_subset.cpu()
        # print(tgt_dict)
        loss += tgt_dict['scale'] * loss_func(pr_subset,
                                              torch.ones(pr_subset.shape, device=pr_subset.device) * tgt_dict['label'])
    return loss


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
