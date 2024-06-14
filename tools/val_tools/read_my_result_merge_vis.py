import json
from func import *
import cv2
import os
from utils.models import metrics
from utils.models.RPHCounter import calculate_tp_fp_fn

"""
加载PestCounter.test.py的.json检测文件，然后计算指标
"""

split = 4
merge_gt_sum = 0
merge_pred_sum = 0
current_dir = os.path.dirname(os.path.abspath(__file__))

# GT label
folder_roots = [
    os.path.join(current_dir, '..', '..', 'data', 'test_data')
]

# Pred result
res_files = [
    os.path.join(current_dir, '..', '..', 'checkpoints', '2024-02-21_10-40-08', 'res.json')
]

val_meter = metrics.Meter()
split_meter = CalSplit()

vis_root = os.path.join(current_dir, '..', '..', 'checkpoints', '2024-02-21_10-40-08', 'vis')
o_gt_path = os.path.join(vis_root, 'o_gt')
o_pred_path = os.path.join(vis_root, 'o_pred')
o_heatmap_path = os.path.join(vis_root, 'o_heatmap')
os.makedirs(o_gt_path, exist_ok=True)
os.makedirs(o_pred_path, exist_ok=True)
os.makedirs(o_heatmap_path, exist_ok=True)
gt_merge = []
pred_merge = []
heatmap_merge = []

for folder_root, res_file in zip(folder_roots, res_files):

    data = read_my_res(res_file)
    for idx, d in enumerate(data):
        img_name = d['image_names']
        print(img_name)

        # 读取可视化的图片，然后在后面把四张小的拼成原来的大图
        origin_name, cut_num = extract_info(img_name)
        gt_merge.append(cv2.imread(os.path.join(vis_root, 'gt', img_name)))
        heatmap_merge.append(cv2.imread(os.path.join(vis_root, 'heatmap', img_name)))

        json_path = os.path.join(folder_root, 'box_label', img_name.replace('jpg', 'json'))
        gt_boxes = get_gt_boxes(json_path)

        pred_points = d['pred_points']

        tp, fp, fn = calculate_tp_fp_fn(gt_boxes, pred_points)
        val_meter.add_tpfpfn(tp, fp, fn)

        fn_boxes = get_fn_boxes(gt_boxes, pred_points)
        pred_img = cv2.imread(os.path.join(vis_root, 'pred', img_name))
        box_on_image(pred_img, fn_boxes, (0, 255, 255))
        pred_merge.append(pred_img)

        gt_sum = len(gt_boxes)
        pred_sum = len(pred_points)

        # 对于大图切分检测的情况的MSE计算，例如分16张，每检测16次才能加一次总误差
        merge_gt_sum += gt_sum
        merge_pred_sum += pred_sum
        if (idx + 1) % split == 0:

            o_gt = cat_img(gt_merge)
            o_pred = cat_img(pred_merge)
            o_heatmap = cat_img(heatmap_merge)

            tangle_text(o_gt, "True num:" + str(merge_gt_sum))
            tangle_text(o_pred, "Pred num:" + str(merge_pred_sum))

            cv2.imwrite(os.path.join(o_gt_path, origin_name + '.jpg'), o_gt)
            cv2.imwrite(os.path.join(o_pred_path, origin_name + '.jpg'), o_pred)
            cv2.imwrite(os.path.join(o_heatmap_path, origin_name + '.jpg'), o_heatmap)
            gt_merge.clear()
            pred_merge.clear()
            heatmap_merge.clear()
            # cv2.imshow('gt', o_gt)
            # cv2.waitKey(0)

            mis_counts = abs(merge_gt_sum - merge_pred_sum)
            print(mis_counts)
            split_meter.add(merge_gt_sum, merge_pred_sum)
            merge_gt_sum, merge_pred_sum = 0, 0

print(val_meter.tp, val_meter.fp, val_meter.fn)
p, r, f = val_meter.get_f1()
val_dict = {'val_precision': p, 'val_recall': r, 'val_f1': f}
print(val_dict)
print(split_meter.get_res())
