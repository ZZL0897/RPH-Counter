import os
import shutil
import random


def split_data(input_folder, output_train_folder, output_val_folder, split_ratio=0.8, seed=None):
    if seed is not None:
        random.seed(seed)

    # 获取文件列表
    json_files = [f for f in os.listdir(input_folder) if f.endswith('.json')]

    # 计算分割点
    split_point = int(len(json_files) * split_ratio)

    # 打乱文件顺序
    random.shuffle(json_files)

    # 创建输出目录
    os.makedirs(output_train_folder, exist_ok=True)
    os.makedirs(output_val_folder, exist_ok=True)

    # 分割文件
    train_files = json_files[:split_point]
    val_files = json_files[split_point:]

    # 移动文件到相应目录
    for file in train_files:
        img_name = file.replace('.json', '.jpg')
        shutil.move(os.path.join(input_folder, file), os.path.join(output_train_folder, file))
        shutil.move(os.path.join(input_folder, img_name), os.path.join(output_train_folder, img_name))

    for file in val_files:
        img_name = file.replace('.json', '.jpg')
        shutil.move(os.path.join(input_folder, file), os.path.join(output_val_folder, file))
        shutil.move(os.path.join(input_folder, img_name),os.path.join(output_val_folder, img_name))


if __name__ == '__main__':
    # 使用示例
    input_folder = r""
    output_train_folder = r""
    output_val_folder = r""
    split_ratio = 0.8

    split_data(input_folder, output_train_folder, output_val_folder, split_ratio=split_ratio, seed=42)
