import json
import os

from natsort import natsorted
import pickle
from utils.models import metrics
import re
import numpy as np
import cv2


def get_gt_boxes(json_path):
    with open(json_path, 'r') as file:
        label_data = json.load(file)
        gt_boxes = []
        shapes = label_data["shapes"]
        for shape in shapes:
            box = shape["points"]
            x1, y1 = box[0]
            x2, y2 = box[1]
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
            gt_boxes.append([[int(x1), int(y1)], [int(x2), int(y2)]])
    return gt_boxes


def read_my_res(res_file):
    with open(res_file, 'rb') as f:
        data = json.load(f)
    data = natsorted(data, key=lambda x: x['image_names'].lower())
    return data


def read_mm_res(res_file):
    with open(res_file, 'rb') as f:
        data = pickle.load(f)
        # print(data)
    data = natsorted(data, key=lambda x: x['img_path'].lower())
    return data


def read_mm2_res(res_file, img_paths, json_files):
    data_alter = []
    with open(res_file, 'rb') as f:
        data = pickle.load(f)
        # print(data)

    # mmdet2的pkl中只有检测的框与概率，前4个元素为框，最后一个是概率
    # 其顺序与os.listdir读取文件名的顺序一致
    for i, d in enumerate(data):
        dic = {'img_path': img_paths[i],
               'gt_json': json_files[i],
               'pred': d}
        data_alter.append(dic)

    data = natsorted(data_alter, key=lambda x: x['img_path'].lower())

    return data

def read_yolo_res(res_file, gt_path):
    from pathlib import Path
    with open(res_file, 'rb') as f:
        data = json.load(f)
        # print(data)
    data = natsorted(data, key=lambda x: x['image_id'].lower())
    data2 = []
    cur = ''
    cur_i = -1
    has_res_img_names = {}
    for d in data:
        image_id = d['image_id']
        if cur != image_id:
            cur = image_id
            has_res_img_names[image_id] = 1
            data2.append({'image_id': cur, 'bboxes': [xywh2xyxy(d['bbox'])], 'scores': [d['score']]})
            cur_i += 1
        elif cur == image_id:
            data2[cur_i]['bboxes'].append(xywh2xyxy(d['bbox']))
            data2[cur_i]['scores'].append(d['score'])
    gt_files = os.listdir(gt_path)

    for f in gt_files:
        image_id = Path(f).stem
        if image_id not in has_res_img_names:
            data2.append({'image_id': image_id, 'bboxes': [], 'scores': []})
    data2 = natsorted(data2, key=lambda x: x['image_id'].lower())

    return data2


def xywh2xyxy(cood):
    x, y, w, h = cood
    x1 = x
    y1 = y
    x2 = x + w - 1
    y2 = y + h - 1
    return [int(x1), int(y1), int(x2), int(y2)]


class CalSplit:
    def __init__(self, low=10, mid=50):
        self.low = low
        self.mid = mid
        self.cal_low = metrics.Meter()
        self.cal_mid = metrics.Meter()
        self.cal_high = metrics.Meter()
        self.cal_total = metrics.Meter()

    def add(self, gt_sum, pred_sum):
        mis_counts = abs(gt_sum - pred_sum)
        if gt_sum <= self.low:
            self.cal_low.add(mis_counts, 1)
        elif self.low < gt_sum <= self.mid:
            self.cal_mid.add(mis_counts, 1)
        elif gt_sum > self.mid:
            self.cal_high.add(mis_counts, 1)
        self.cal_total.add(mis_counts, 1)

    def get_res(self):
        low_mae, low_mse = self.cal_low.get_avg_score(), self.cal_low.get_mse()
        mid_mae, mid_mse = self.cal_mid.get_avg_score(), self.cal_mid.get_mse()
        high_mae, high_mse = self.cal_high.get_avg_score(), self.cal_high.get_mse()
        total_mae, total_mse = self.cal_total.get_avg_score(), self.cal_total.get_mse()
        res = {str(self.low)+'_mae': low_mae,
               str(self.low)+'_mse': low_mse,
               str(self.low) + '-' + str(self.mid) + '_mae': mid_mae,
               str(self.low) + '-' + str(self.mid) + '_mse': mid_mse,
               str(self.mid) + '_mae': high_mae,
               str(self.mid) + '_mse': high_mse,
               'total_mae': total_mae,
               'total_mse': total_mse
               }
        return res


def extract_info(file_name):
    # 定义正则表达式模式
    pattern = re.compile(r'(.+)_rectangle_(\d+)\.jpg')

    # 使用正则表达式匹配文件名
    match = pattern.match(file_name)

    if match:
        original_name = match.group(1)  # 获取原文件名称
        cut_number = int(match.group(2))  # 获取切割编号并转换为整数
        return original_name, cut_number
    else:
        # 如果匹配失败，返回默认值或者抛出异常，取决于你的需求
        return None, None


def cat_img(img_list, normalize=False):
    top_row = np.concatenate((img_list[0], img_list[1]), axis=1)
    bottom_row = np.concatenate((img_list[2], img_list[3]), axis=1)
    final_image = np.concatenate((top_row, bottom_row), axis=0)

    if normalize:
        min_val = np.min(final_image)
        max_val = np.max(final_image)
        final_image = (final_image - min_val) / (max_val - min_val)
        final_image = cv2.applyColorMap(np.uint8(final_image * 255.0), cv2.COLORMAP_JET)

    return final_image


def tangle_text(img, text):
    font = cv2.FONT_HERSHEY_TRIPLEX
    font_size = 5
    font_color = (255, 255, 255)
    thickness = 6
    # 计算文本框大小
    text_size = cv2.getTextSize(text, font, font_size, thickness)[0]
    rect_size = (text_size[0] + 20, text_size[1] + 50)  # 加上边距
    # 绘制矩形和文本
    cv2.rectangle(img, (0, 0), (rect_size[0], rect_size[1]), (240, 170, 0), thickness=-1)
    # 计算矩形框中心点坐标
    rect_center = (rect_size[0] // 2, rect_size[1] // 2)
    # 计算文本起始坐标，确保文本相对于矩形框居中
    text_start = (rect_center[0] - text_size[0] // 2, rect_center[1] + text_size[1] // 2)
    # 绘制文本
    cv2.putText(img, text, text_start, font, font_size, font_color, thickness)
    # return img


def box_on_image(image, boxes, bgr=(0, 0, 255)):
    for box in boxes:
        x1, y1 = box[0]
        x2, y2 = box[1]
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), bgr, 6)


def point_in_rectangle(point, rectangle):
    x, y = point
    x1, y1 = rectangle[0]
    x2, y2 = rectangle[1]
    # print(x, y)
    # print(x1, x2, y1, y2)

    return x1 <= x <= x2 and y1 <= y <= y2


def get_fn_boxes(rectangles, points):
    _rectangles = rectangles.copy()
    _points = points.copy()
    for point in _points:
        for rectangle in _rectangles[:]:  # Iterate over a copy of rectangles
            if point_in_rectangle(point, rectangle):
                _rectangles.remove(rectangle)
                break

    return _rectangles


def get_tp_fp_fn_boxes(rectangles, pred_boxes, points):
    _rectangles = rectangles.copy()
    _points = points.copy()
    _pred_boxes = pred_boxes.copy()
    tp_boxes = []
    fp_boxes = []
    for i, point in enumerate(_points):
        for j, rectangle in enumerate(_rectangles[:]):  # Iterate over a copy of rectangles
            if point_in_rectangle(point, rectangle):
                _rectangles.remove(rectangle)
                tp_boxes.append(pred_boxes[i])
                break
        else:
            fp_boxes.append(pred_boxes[i])
    fn_boxes = _rectangles

    return tp_boxes, fp_boxes, fn_boxes


class R2:
    def __init__(self):
        self.pred_list = []
        self.gt_list = []

    def add(self, pred, gt):
        self.pred_list.append(pred)
        self.gt_list.append(gt)

    def get_r2(self):
        from sklearn.metrics import r2_score
        r2 = r2_score(self.gt_list, self.pred_list)
        print(r2)

