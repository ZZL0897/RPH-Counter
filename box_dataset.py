import cv2
from torch.utils import data
import glob
import numpy as np
import torch
import os
from skimage.io import imread
import torchvision.transforms.functional as FT
from torch.utils.data.dataset import T_co
import collections
from torchvision import transforms
import json
from torch.utils.data import DataLoader
import random


class BoxDataset(data.Dataset):
    def __init__(self, split, datadir, exp_dict, aug_prob=0):
        self.split = split
        self.n_classes = 1
        self.aug_prob = aug_prob

        if split == "train":
            self.path = os.path.join(datadir, "train_data_half")
            self.img_names = [os.path.basename(x) for x in
                              glob.glob(os.path.join(self.path, "images", "*"))]

        # 这个文件夹存放切了16张之后，只包含有虫的图，训练的时候用
        elif split == "val":
            self.path = os.path.join(datadir, "test_data")
            self.img_names = [os.path.basename(x) for x in
                              glob.glob(os.path.join(self.path, "images", "*"))]

        # 这个文件夹存放基于所有原图切成16张的图片，最后验证的时候用
        else:
            self.path = os.path.join(datadir, split)
            self.img_names = [os.path.basename(x) for x in
                              glob.glob(os.path.join(self.path, "images", "*"))]

        torch.multiprocessing.set_sharing_strategy('file_system')

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index) -> T_co:
        name = self.img_names[index]

        # LOAD IMG, POINT, and ROI
        image_path = os.path.join(self.path, "images", name)
        image = imread(image_path)
        if image.ndim == 2:
            image = image[:, :, None].repeat(3, 2)

        label_path = image_path.replace('images', 'box_label').replace('.jpg', '.json')

        r = random.uniform(0, 1)
        threshold = self.aug_prob

        if r < threshold and self.split == 'train':
            image = cv2.flip(image, 1)

        with open(label_path, 'r') as file:
            label_data = json.load(file)
            centers = []
            boxes = []
            shapes = label_data["shapes"]

            flag = 0
            for shape in shapes:
                if r < threshold and self.split == 'train':
                    for i, point in enumerate(shape['points']):
                        point[0] = image.shape[1] - point[0]
                        # 这样就保证了翻转之后box坐标仍是左上+右下的布局
                        if flag == 1:
                            shape['points'][i - 1][0], shape['points'][i][0] = shape['points'][i][0], \
                                                                               shape['points'][i - 1][0]
                            flag = 0
                        else:
                            flag += 1

                box = shape["points"]
                x1, y1 = box[0]
                x2, y2 = box[1]
                if x1 > x2:
                    x1, x2 = x2, x1
                if y1 > y2:
                    y1, y2 = y2, y1
                centers.append(find_rect_centers(x1, x2, y1, y2, rand_offset=True))
                boxes.append([[int(x1), int(y1)], [int(x2), int(y2)]])
            pointList = np.array(centers)

        # for box in boxes:
        #     x1, y1 = box[0]
        #     x2, y2 = box[1]
        #     print(x1, y1, x2, y2)
        #     cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        # cv2.imshow('o', cv2.resize(image, (864, 1152)))
        # cv2.waitKey(0)

        points = np.zeros(image.shape[:2], "uint8")[:, :, None]
        H, W = image.shape[:2]
        for x, y in pointList:
            points[min(int(y), H - 1), min(int(x), W - 1)] = 1

        counts = torch.LongTensor(np.array([pointList.shape[0]]))

        # collection = list(map(FT.to_pil_image, [image, points]))
        image, points = apply_transform(image, points)

        r = {"images": image,
             "points": points.squeeze(),
             "counts": counts,
             'meta': {"index": index}}

        # 训练时需要label box边界
        if self.split == 'train':
            bool_array = gen_boundary(boxes, H, W, rand_offset=True).ravel()
            r['boundaries'] = bool_array
        else:
            r['boxes'] = boxes
        # print(r)
        # print(points.shape)
        # print(points.squeeze().shape)
        return r


def gen_boundary(boxes, h, w, rand_offset=False):
    bool_array = np.zeros((h, w), dtype=bool)

    for box in boxes:
        x1, y1 = box[0]
        x2, y2 = box[1]

        # 为每个边生成随机偏移量
        top_offset = np.random.randint(-2, 3)
        bottom_offset = np.random.randint(-2, 3)
        left_offset = np.random.randint(-2, 3)
        right_offset = np.random.randint(-2, 3)

        if rand_offset:
            y1 += top_offset
            y2 += bottom_offset
            x1 += left_offset
            x2 += right_offset

        # 调整边缘坐标，确保在图像范围内
        y1 = int(max(0, min(y1, bool_array.shape[0] - 1)))
        y2 = int(max(0, min(y2, bool_array.shape[0] - 1)))
        x1 = int(max(0, min(x1, bool_array.shape[1] - 1)))
        x2 = int(max(0, min(x2, bool_array.shape[1] - 1)))

        # 修改矩形边缘处的元素值为1，考虑了随机偏移量
        bool_array[y1, x1:x2 + 1] = True  # 上边缘
        bool_array[y2, x1:x2 + 1] = True  # 下边缘
        bool_array[y1:y2 + 1, x1] = True  # 左边缘
        bool_array[y1:y2 + 1, x2] = True  # 右边缘

    return bool_array


def find_rect_centers(x1, x2, y1, y2, rand_offset=False):
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    if rand_offset:
        # 生成随机偏移量
        offset_x = np.random.randint(-2, 3)  # 在[-2, -1, 0, 1, 2]中选择一个整数
        offset_y = np.random.randint(-2, 3)
        # 计算新中心
        center_x += offset_x
        center_y += offset_y
    return center_x, center_y


def apply_transform(image, points):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform = ComposeJoint(
        [
            [transforms.ToTensor(), None],
            [transforms.Normalize(mean=mean, std=std), None],
            [None, ToLong()]
        ])

    return transform([image, points])


class ComposeJoint(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for transform in self.transforms:
            x = self._iterate_transforms(transform, x)

        return x

    def _iterate_transforms(self, transforms, x):
        if isinstance(transforms, collections.Iterable):
            for i, transform in enumerate(transforms):
                x[i] = self._iterate_transforms(transform, x[i])
        else:

            if transforms is not None:
                x = transforms(x)

        return x


class ToLong(object):
    def __call__(self, x):
        return torch.LongTensor(np.asarray(x))


if __name__ == '__main__':
    dataset = BoxDataset('train', r'G:\ph_data2', None)
    data_loder = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)
    # print(len(data_loder))
    for batch in data_loder:
       i = batch['meta']['index']
       print(dataset.img_names[i])
