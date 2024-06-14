import os.path

import cv2
import matplotlib.cm as CM
import numpy as np
import torch
import tqdm
from PIL import Image
from PIL import ImageDraw
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from utils import losses
from . import base_networks, metrics


class RPHCounter(torch.nn.Module):
    def __init__(self, exp_dict, train_set):
        super().__init__()
        self.exp_dict = exp_dict
        self.n_classes = train_set.n_classes

        self.model_base = base_networks.get_base(self.exp_dict['model'],
                                                 self.exp_dict, n_classes=self.n_classes)

        if self.exp_dict["optimizer"] == "adam":
            self.opt = torch.optim.Adam(
                self.model_base.parameters(), lr=self.exp_dict["lr"], betas=(0.99, 0.999), weight_decay=0.0001)
            # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.opt, milestones=[15], gamma=0.5)

        elif self.exp_dict["optimizer"] == "sgd":
            self.opt = torch.optim.SGD(
                self.model_base.parameters(), lr=self.exp_dict["lr"], momentum=0.9, weight_decay=0.0001, nesterov=True)
            # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.opt, milestones=[15], gamma=0.5)

        else:
            raise ValueError

    def train_on_loader(model, train_loader):
        model.train()

        batch_size = train_loader.batch_size
        n_batches = len(train_loader)
        train_meter = metrics.Meter()

        pbar = tqdm.tqdm(total=n_batches, position=0, ncols=80)
        for i, batch in enumerate(train_loader):
            loss_on_batch = model.train_on_batch(batch)
            # loss_on_batch = model.train_on_batch_accumulation(batch, i)
            train_meter.add_loss(*loss_on_batch, batch_size)

            pbar.set_description("Training. Loss: %.4f" % train_meter.get_avg_score())
            pbar.update(1)

        pbar.close()
        # model.scheduler.step()
        train_loss, img_loss, point_loss, boundary_loss, fp_loss = train_meter.get_avg_loss()
        return {'train_loss': train_loss,
                'image_loss': img_loss,
                'point_loss': point_loss,
                'boundary_loss': boundary_loss,
                'fp_loss': fp_loss,
                'learning_rate': model.opt.state_dict()['param_groups'][0]['lr']}

    def train_on_batch(self, batch):
        self.opt.zero_grad()
        self.train()

        images = batch["images"].cuda()
        points = batch["points"].long().cuda()
        boundaries = batch["boundaries"].cuda()
        logits = self.model_base.forward(images)

        # 计算总损失，其中probs是网络输出的预测，由于是全卷积网络且只有一个类别所以用sigmoid激活
        img_loss, point_loss, boundary_loss, fp_loss = losses.compute_total_loss(points=points,
                                                                                 probs=logits.sigmoid(),
                                                                                 boundaries=boundaries)
        loss = img_loss + point_loss + boundary_loss + fp_loss
        loss.backward()
        self.opt.step()

        return losses.tensor2float(loss), \
               losses.tensor2float(img_loss), \
               losses.tensor2float(point_loss), \
               losses.tensor2float(boundary_loss), \
               losses.tensor2float(fp_loss)

    # 对于显存不够的情况使用梯度累加，使实际batchsize大小等价于accumulation * bs
    def train_on_batch_accumulation(self, batch, i, accumulation=16):
        # self.opt.zero_grad()
        self.train()

        images = batch["images"].cuda()
        points = batch["points"].long().cuda()
        logits = self.model_base.forward(images)

        # 计算总损失，其中probs是网络输出的预测，由于是全卷积网络且只有一个类别所以用sigmoid激活
        img_loss, point_loss, boundary_loss, fp_loss = losses.compute_total_loss(points=points,
                                                                                 probs=logits.sigmoid(),
                                                                                 boundaries=batch["boundaries"])
        loss = img_loss + point_loss + boundary_loss + fp_loss
        train_loss = loss.item()
        loss = loss / accumulation
        loss.backward()

        if (i + 1) % accumulation == 0:
            # optimizer the net
            self.opt.step()  # update parameters of net
            self.opt.zero_grad()  # reset gradient

        return losses.tensor2float(loss), \
               losses.tensor2float(img_loss), \
               losses.tensor2float(point_loss), \
               losses.tensor2float(boundary_loss), \
               losses.tensor2float(fp_loss)

    @torch.no_grad()
    def val_on_loader(self, val_loader, img_names=None, savedir=None, n_images=2):
        self.eval()

        n_batches = len(val_loader)
        val_meter = metrics.Meter()
        result = []
        pbar = tqdm.tqdm(total=n_batches, position=0, ncols=80)
        for i, batch in enumerate(val_loader):
            score_dict, pred_points_list = self.val_on_batch(batch)

            val_meter.add(score_dict['miscounts'], batch['images'].shape[0])
            val_meter.add_tpfpfn(score_dict['tp'], score_dict['fp'], score_dict['fn'])

            # 测试看那些图片效果较差
            # tp, fp, fn = score_dict['tp'], score_dict['fp'], score_dict['fn']
            # if tp + fn != 0 and tp + fp != 0:
            #     recall = tp / (tp + fn)
            #     precision = tp / (tp + fp)
            #     f1 = (2 * precision * recall) / (precision + recall)
            #     # if f1 < 0.8:
            # print(img_names[batch['meta']['index']])
            # print(tp, fp, fn)

            if img_names is not None:
                img_name = img_names[batch['meta']['index']]
                result.append({'image_names': img_name, 'pred_points': pred_points_list})

            pbar.update(1)

            # if score_dict['miscounts'] >= 10:
            if savedir and i < n_images:
                # os.makedirs(savedir_images, exist_ok=True)
                self.vis_on_batch(batch, savedir=savedir, image_name=img_names[batch['meta']['index']])

                p, r, f = val_meter.get_f1()
                # pbar.set_description(
                #     "Val MAE:{} MSE: {} Precision: {} Recall: {} F1: {}".format(val_meter.get_avg_score(),
                #                                                                 val_meter.get_mse(),
                #                                                                 p, r, f))

        pbar.close()
        val_mae = val_meter.get_avg_score()
        val_mse = val_meter.get_mse()
        p, r, f = val_meter.get_f1()
        val_dict = {'val_mae': val_mae, 'val_mse': val_mse, 'val_precision': p, 'val_recall': r, 'val_f1': f}
        return val_dict, result

    def val_on_batch(self, batch):
        # 如果val的batch大于1，下面的指标计算会出问题
        self.eval()
        images = batch["images"].cuda()
        points = batch["points"].long().cuda()
        boxes = batch["boxes"]
        logits = self.model_base.forward(images)
        # p = logits.softmax()
        probs = logits.sigmoid().cpu().numpy()

        blobs = losses.get_blobs(probs=probs)

        pred_points_list = losses.blobs2points_list(blobs)
        # print(pred_points_list, boxes)
        tp, fp, fn = calculate_tp_fp_fn(boxes, pred_points_list)
        # print(tp, fp, fn)

        return {'miscounts': abs(float((np.unique(blobs) != 0).sum() - (points != 0).sum())),
                'tp': tp, 'fp': fp, 'fn': fn}, pred_points_list

    @torch.no_grad()
    def vis_on_batch(self, batch, savedir, image_name):
        self.eval()
        images = batch["images"].cuda()
        points = batch["points"].long().cuda()
        boxes = batch["boxes"]
        logits = self.model_base.forward(images)
        probs = logits.sigmoid().cpu().numpy()

        blobs = losses.get_blobs(probs=probs)

        pred_counts = (np.unique(blobs) != 0).sum()
        pred_blobs = blobs
        pred_probs = probs.squeeze()

        # loc
        pred_count = pred_counts.ravel()[0]
        pred_blobs = pred_blobs.squeeze()

        # img_org = hu.get_image(batch["images"],denorm="rgb")
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img_org = (batch["images"] * torch.tensor(std).view(3, 1, 1)) + torch.tensor(mean).view(3, 1, 1)
        img_org = np.ascontiguousarray(np.uint8(img_org.squeeze(0).permute(1, 2, 0).numpy() * 255))
        # cv2.imshow('test', img_org)
        # cv2.waitKey(0)

        img = img_org.copy()

        # true points
        y_list, x_list = np.where(batch["points"][0].long().numpy().squeeze())
        # 将box, GT点展示在原图上
        box_on_image(img_org, boxes)
        img_peaks = points_on_image(y_list, x_list, img_org)
        # text = "%s ground truth" % (batch["points"].sum().item())
        # text_on_image(text=text, image=img_peaks)
        # cv2.imshow('test', img_org)
        # cv2.waitKey(0)
        create_dir(os.path.join(savedir, 'gt'))
        cv2.imwrite(os.path.join(savedir, 'gt', image_name), cv2.cvtColor((img_peaks * 255).astype('uint8'), cv2.COLOR_RGB2BGR))

        # pred points
        pred_points = losses.blobs2points(pred_blobs).squeeze()
        pred_y_list, pred_x_list = np.where(pred_points.squeeze())
        # img_pred = mask_on_image(img, pred_blobs)
        # img_pred = points_on_image(pred_y_list, pred_x_list, img_org)
        img_pred = vis_tp_fp_on_img(blobs, points, img)
        # text = "%s predicted" % (len(pred_y_list))
        # text_on_image(text=text, image=img_pred)
        # cv2.imshow('gt', cv2.cvtColor(cv2.resize(img_peaks * 255, (768, 1024)).astype('uint8'), cv2.COLOR_RGB2BGR))
        # cv2.imshow('pred', cv2.cvtColor(cv2.resize(img_pred * 255, (768, 1024)).astype('uint8'), cv2.COLOR_RGB2BGR))
        # cv2.waitKey(0)
        create_dir(os.path.join(savedir, 'pred'))
        cv2.imwrite(os.path.join(savedir, 'pred', image_name), cv2.cvtColor((img_pred * 255).astype('uint8'), cv2.COLOR_RGB2BGR))

        heatmap = cv2.applyColorMap(np.uint8(pred_probs * 255.0), cv2.COLORMAP_JET)
        # # cv2.imshow('heatmap', cv2.resize(heatmap, (768, 1024)))
        # # cv2.waitKey(0)
        create_dir(os.path.join(savedir, 'heatmap'))
        cv2.imwrite(os.path.join(savedir, 'heatmap', image_name), heatmap)

    # def get_state_dict(self):
    #     state_dict = {"model": self.model_base.state_dict(),
    #                   "opt": self.opt.state_dict()}
    #
    #     return state_dict
    #
    # def load_state_dict(self, state_dict):
    #     self.model_base.load_state_dict(state_dict["model"])
    #     self.opt.load_state_dict(state_dict["opt"])


def create_dir(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def points_on_image(y_list, x_list, image, radius=3, c_list=None):
    image_uint8 = image
    H, W, _ = image_uint8.shape
    color_list = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    for i, (y, x) in enumerate(zip(y_list, x_list)):
        if y < 1:
            x, y = int(x * W), int(y * H)
        else:
            x, y = int(x), int(y)

        # Blue color in BGR
        if c_list is not None:
            color = color_list[c_list[i]]
        else:
            color = color_list[1]

        # Line thickness of 2 px
        thickness = 3
        # Using cv2.rectangle() method
        # Draw a rectangle with blue line borders of thickness of 2 px
        image_uint8 = cv2.circle(image_uint8, (x, y), radius, (0, 255, 0), thickness)

        # start_point = (x - radius * 2, y - radius * 2)
        # end_point = (x + radius * 2, y + radius * 2)
        # thickness = 2
        # color = (255, 0, 0)
        #
        # image_uint8 = cv2.rectangle(image_uint8, start_point, end_point, color, thickness)

    return image_uint8 / 255.0


def points_on_image2(y_list, x_list, image, rgb=(0, 255, 0), alpha=1.0):
    image_uint8 = image.copy().astype(dtype=np.uint8)
    rgb = [c+1 if c < 255 else c for c in rgb]
    H, W, _ = image_uint8.shape
    blk = np.zeros_like(image_uint8, np.uint8)
    for i, (y, x) in enumerate(zip(y_list, x_list)):
        if y < 1:
            x, y = int(x * W), int(y * H)
        else:
            x, y = int(x), int(y)
        thickness = 6
        radius = 12
        # Using cv2.rectangle() method
        # Draw a rectangle with blue line borders of thickness of 2 px
        blk = cv2.circle(blk, (x, y), radius, rgb, thickness)

        thickness = 6
        radius = 6
        start_point = (x - radius * 2, y - radius * 2)
        end_point = (x + radius * 2, y + radius * 2)

        blk = cv2.rectangle(blk, start_point, end_point, rgb, thickness)

    obj_ids = np.unique(blk)
    for o in obj_ids:
        if o == 0:
            continue
        ind = blk == o
        image_uint8[ind] = image_uint8[ind] * (1 - alpha) + blk[ind] * alpha

    return image_uint8 / 255.0


def text_on_image(text, image):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 40)
    fontScale = 0.8
    fontColor = (1, 1, 1)
    lineType = 1
    # img_mask = skimage.transform.rescale(np.array(img_mask), 1.0)
    # img_np = skimage.transform.rescale(np.array(img_points), 1.0)
    img_np = cv2.putText(
        image,
        text,
        bottomLeftCornerOfText,
        font,
        fontScale,
        fontColor,
        thickness=2
        # lineType
    )
    return img_np


def mask_on_image(image, mask, rgb=(0, 255, 0), alpha=0.1, return_pil=False):
    mask = np.array(mask).squeeze()
    obj_ids = np.unique(mask)

    color = np.zeros(image.shape, dtype="uint8")
    color[:, :, 0] = rgb[0]
    color[:, :, 1] = rgb[1]
    color[:, :, 2] = rgb[2]

    result = image.copy()
    for o in obj_ids:
        if o == 0:
            continue
        ind = mask == o
        result[ind] = result[ind] * alpha + color[ind] * (1 - alpha)
    # result = mark_boundaries(result, mask)

    if return_pil:
        return Image.fromarray(result)

    return result / 255.0


def box_on_image(image, boxes):
    for box in boxes:
        x1, y1 = box[0]
        x2, y2 = box[1]
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 5)


def point_in_rectangle(point, rectangle):
    x, y = point
    x1, y1 = rectangle[0]
    x2, y2 = rectangle[1]
    # print(x, y)
    # print(x1, x2, y1, y2)

    return x1 <= x <= x2 and y1 <= y <= y2


def vis_tp_fp_on_img(blobs, points, image):
    blobs_t = torch.from_numpy(blobs).cuda()
    u_list = torch.unique(blobs_t)
    tp_mask = torch.zeros_like(blobs_t)
    fp_mask = torch.zeros_like(blobs_t)
    more_mask = torch.zeros_like(blobs_t)
    for n in u_list:
        if n == 0:
            continue
        blobs_u = (blobs_t == n).int()
        has_points = points * blobs_u
        num_points_in_blob = torch.sum(has_points)
        blobs_u[blobs_u == 1] = n
        if num_points_in_blob == 1:
            tp_mask += blobs_u
        elif num_points_in_blob == 0:
            fp_mask += blobs_u
        else:
            more_mask += blobs_u
    # i1 = mask_on_image(image, tp_mask.cpu().numpy(), (0, 255, 40), alpha=0.1)
    # i2 = mask_on_image(i1*255.0, fp_mask.cpu().numpy(), (200, 0, 30), alpha=0.3)
    i3 = mask_on_image(image, more_mask.cpu().numpy(), (0, 0, 255), alpha=0.3)

    tp_points = losses.blobs2points(tp_mask.cpu().numpy()).squeeze()
    fp_points = losses.blobs2points(fp_mask.cpu().numpy()).squeeze()

    t_y, t_x = np.where(tp_points.squeeze())
    it = points_on_image2(t_y, t_x, i3*255.0, (0, 255, 40))
    f_y, f_x = np.where(fp_points.squeeze())
    i_f = points_on_image2(f_y, f_x, it*255.0, (200, 0, 30), alpha=0.7)

    return i_f


# old code
# def calculate_tp_fp_fn(rectangles, points):
#     tp, fp, fn = 0., 0., 0.
#
#     for point in points:
#         point_in_any_rectangle = any(point_in_rectangle(point, rectangle) for rectangle in rectangles)
#         # print(point_in_any_rectangle)
#
#         if point_in_any_rectangle:
#             tp += 1
#         else:
#             fp += 1
#
#         # 计算 FN
#     fn = len(rectangles) - tp
#
#     return tp, fp, fn


def calculate_tp_fp_fn(rectangles, points):
    tp, fp, fn = 0, 0, 0
    _rectangles = rectangles.copy()
    _points = points.copy()
    for point in _points:
        for rectangle in _rectangles[:]:  # Iterate over a copy of rectangles
            if point_in_rectangle(point, rectangle):
                tp += 1
                _rectangles.remove(rectangle)
                break
        else:
            fp += 1

    # 计算 FN
    fn = len(_rectangles)

    return tp, fp, fn
