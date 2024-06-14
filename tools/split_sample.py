import json
import os
import random
from PIL import Image
import cv2
import torch

# 全局环境变量 策略
# strategy = "old"
# strategy = "4"
strategy = "4"


# strategy = "1"


# 构建label.json
def label_json_builder(shapes, image_path, image_size=(864, 1152)):
    data = {
        "version": "0.3.3",
        "flags": {},
        "shapes": shapes,
        "imagePath": os.path.basename(image_path),
        "imageData": None,  # You can modify this if needed
        "imageHeight": image_size[1],
        "imageWidth": image_size[0]
    }

    json_data = json.dumps(data, indent=2)  # Convert data to JSON string with indentation
    return json_data


# 计算矩形的最左上角和最右下角的点，用于进一步计算最小外接矩形
def calculate_min_outside_rectangles(rectangles):
    min_x, min_y, max_x, max_y = None, None, None, None
    # 查找最左上角和最右下角的点
    for i, rectangle in enumerate(rectangles, 1):
        points = rectangle["points"]

        for point in points:
            x, y = point
            if min_x is None or x < min_x:
                min_x = x
            if min_y is None or y < min_y:
                min_y = y
            if max_x is None or x > max_x:
                max_x = x
            if max_y is None or y > max_y:
                max_y = y

    return [min_x, min_y], [max_x, max_y]


# 获得16个的矩形
def get_split_16_rect(raw_size=(3456, 4608)):
    rects = []

    raw_width, raw_height = raw_size
    cell_width = raw_width // 4
    cell_height = raw_height // 4
    for i in range(4):
        for j in range(4):
            left_top_x = i * cell_width
            left_top_y = j * cell_height
            right_bottom_x = left_top_x + cell_width
            right_bottom_y = left_top_y + cell_height
            rects.append([left_top_x, left_top_y, right_bottom_x, right_bottom_y])
    return rects


# 获得4个的矩形
def get_split_4_rect(raw_size=(1728, 2304)):
    rectangles = []
    for i in range(4):
        left_top_x = int(raw_size[0] * (i % 2) / 2)
        left_top_y = int(raw_size[1] * (i // 2) / 2)
        right_bottom_x = int(raw_size[0] * (i % 2 + 1) / 2)
        right_bottom_y = int(raw_size[1] * (i // 2 + 1) / 2)
        rectangles.append([left_top_x, left_top_y, right_bottom_x, right_bottom_y])
    return rectangles


# 获得1个的矩形
def get_split_1_rect(raw_size=(864, 1152)):
    rectangles = []
    left_top_x = 0
    left_top_y = 0
    right_bottom_x = raw_size[0]
    right_bottom_y = raw_size[1]
    rectangles.append([left_top_x, left_top_y, right_bottom_x, right_bottom_y])
    return rectangles


# 在raw_image_height*raw_image_width的图像中,随即裁剪一个1152*864大小的矩形

def random_crop_rectangles(raw_image_height, raw_image_width, crop_size=(864, 1152)):
    left_top_point_y_max, left_top_point_x_max = raw_image_height - crop_size[0], raw_image_width - crop_size[1]
    left_top_x = random.randint(0, left_top_point_x_max)
    left_top_y = random.randint(0, left_top_point_y_max)
    return left_top_x, left_top_y, left_top_x + crop_size[1], left_top_y + crop_size[0]


# 从 rectangles 中随机选出 nums_to_crop 个矩形框，然后以每个框为中心，截取nums_to_crop张图片
def random_crop_rectangles_with_list(rectangles, nums_to_crop, crop_size=(864, 1152)):
    center_rect_list = random.sample(rectangles, nums_to_crop)
    crop_rectangles = []
    for center_rect in center_rect_list:
        center_x, center_rect_y = center_rect["points"][0]
        crop_rectangles.append(
            [center_x - crop_size[1] / 2, center_rect_y - crop_size[0] / 2, center_x + crop_size[1] / 2,
             center_rect_y + crop_size[0] / 2])
    return crop_rectangles


# 根据给定的左上角和右下角坐标，生成M*N个矩形，并将这些矩形的左上和右下坐标放入rectangles列表中
def generate_min_outside_rectangles_auto(left_top_point, right_bottom_point, rectangle_size=(864, 1152)):
    # 计算给定矩形的宽度和高度
    total_width = right_bottom_point[0] - left_top_point[0]
    total_height = right_bottom_point[1] - left_top_point[1]

    # 计算每个矩形的宽度和高度
    rectangle_width = rectangle_size[0]
    rectangle_height = rectangle_size[1]

    # print(f"Total width: {total_width}, total height: {total_height}")

    # 计算水平和垂直方向上的间隔
    horizontal_interval = max(rectangle_width, (total_width % rectangle_width) / 2)
    vertical_interval = max(rectangle_height, (total_height % rectangle_height) / 2)

    # 计算自动调整的M和N
    M = max(1, int(total_height / rectangle_height))
    N = max(1, int(total_width / rectangle_width))

    rectangles = []

    # 生成M*N个矩形的左上和右下坐标
    for i in range(M):
        for j in range(N):
            left_top_x = left_top_point[0] + j * horizontal_interval + (horizontal_interval - rectangle_width) / 2
            left_top_y = left_top_point[1] + i * vertical_interval + (vertical_interval - rectangle_height) / 2
            right_bottom_x = left_top_x + rectangle_width
            right_bottom_y = left_top_y + rectangle_height

            rectangles.append([left_top_x, left_top_y, right_bottom_x, right_bottom_y])

    return rectangles, M, N


# 根据限制
def adjust_xy(limits, n):
    if n < 0:
        return 0
    elif n >= limits:
        return limits
    else:
        return n


# 截取矩形区域图像和标签并保存
def crop_and_save_rectangles(image_path, rectangles, output_folder, file_name, json_filename, resize_factor=1,
                             target_shape=(864, 1152)):
    """
    :param image_path: 原始图像路径
    :param rectangles: 矩形区域列表
    :param output_folder: 输出文件夹
    :param file_name: 输出文件名
    :param json_filename: 原始json文件路径
    :param resize_factor: 图像缩放因子
    :param target_shape: 目标图像大小
    :return:
    """

    # 打开原始图像
    # original_image = Image.open(image_path)
    # if resize_factor!= 1:
    #     original_image = original_image.resize((int(original_image.size[0] * resize_factor), int(original_image.size[1] * resize_factor)))
    #     print(f"resize : {original_image.size}")

    cv2_image = cv2.imread(image_path)
    if resize_factor != 1:
        cv2_image = cv2.resize(cv2_image,
                               (int(cv2_image.shape[1] * resize_factor), int(cv2_image.shape[0] * resize_factor)))
        # print(f"cv2 resize : {cv2_image.shape}")
    if json_filename:
        json_data = open(json_filename, "r").read()
        data = json.loads(json_data)
        shape_list = data["shapes"]
        # 对shape_list内的所有坐标进行缩放
        if resize_factor != 1:
            for s in shape_list:
                points = s["points"]
                points[0] = [int(points[0][0] * resize_factor), int(points[0][1] * resize_factor)]
                points[1] = [int(points[1][0] * resize_factor), int(points[1][1] * resize_factor)]
    else:
        shape_list = None
    # 逐个截取矩形区域并保存
    if rectangles is None or len(rectangles) == 0:
        return
    for i in range(len(rectangles)):
        rect = rectangles[i]
        idx = i
        # print(rect)
        left, top, right, bottom = map(int, rect)
        # cropped_image = original_image.crop((left, top, right, bottom))
        cropped_image = cv2_image[top:bottom, left:right]

        # 保存截取的图像
        output_path = f"{output_folder}/{file_name}_rectangle_{idx}.jpg"
        output_json_path = f"{output_folder}/{file_name}_rectangle_{idx}.json"
        shapes = []
        if shape_list:
            for s in shape_list:
                points = s["points"]
                # points[0]和points[1]是矩形框的左上角和右下角
                # 计算矩形框的中心点
                p2 = [int((points[0][0] + points[1][0]) / 2), int((points[0][1] + points[1][1]) / 2)]
                # 如果p2在截取的矩形区域内，则保留该矩形框
                if left <= p2[0] <= right and top <= p2[1] <= bottom:
                    # print("p2 {} in rect {} ".format(str(p2) , str(rect)) )

                    # 先减去rect的左上角，再存入shapes,
                    shapes.append(
                        {"label": s["label"],
                         "points": [
                             (adjust_xy(target_shape[0], (p[0] - left)), adjust_xy(target_shape[1], (p[1] - top)))
                             for p in points], "group_id": None,
                         "shape_type": "rectangle", "flags": {}})

        json_content = label_json_builder(shapes, output_path)
        # 保存json文件
        if not os.path.exists(output_json_path):
            with open(output_json_path, "w") as f:
                f.write(json_content)
        # print("-----------------------------------------------------------")
        # cropped_image.save(output_path)
        cv2.imwrite(output_path, cropped_image)

    # for idx, rect in enumerate(rectangles):


# 老的随即裁剪生成策略
def generate_strategy_control(json_path):
    # 从文件读取json
    json_data = open(json_path, "r").read()
    data = json.loads(json_data)

    # 提取矩形框信息
    rectangles = data["shapes"]
    raw_iamge_height = data["imageHeight"]
    raw_iamge_width = data["imageWidth"]
    # 所有的point坐标转换为整数
    for rectangle in rectangles:
        points = rectangle["points"]
        points[0] = [int(points[0][0]), int(points[0][1])]
        points[1] = [int(points[1][0]), int(points[1][1])]

    num_rectangles = len(rectangles)
    crop_rectangles = []
    # 应用于没有矩形标签，随即两张图片
    if num_rectangles == 0:
        for i in range(2):
            left_top_x, left_top_y, right_bottom_x, right_bottom_y = random_crop_rectangles(raw_iamge_height,
                                                                                            raw_iamge_width)
            crop_rectangles.append([left_top_x, left_top_y, right_bottom_x, right_bottom_y])
    elif num_rectangles == 1:
        crop_rectangles = random_crop_rectangles_with_list(rectangles, 1)
    # 应用于只有一到四个矩形标签，以每个框为中心，截取2张图片
    elif num_rectangles < 4:
        crop_rectangles = random_crop_rectangles_with_list(rectangles, 2)
    # 应用于有五到九个矩形标签，随机选4个框为中心，各截4张
    elif num_rectangles < 10:
        crop_rectangles = random_crop_rectangles_with_list(rectangles, 4)
    # 应用于十个以上个矩形标签，最小外接M*N方框策略
    else:
        left_top_point, right_bottom_point = calculate_min_outside_rectangles(rectangles)
        # 生成矩形
        crop_rectangles, M, N = generate_min_outside_rectangles_auto(left_top_point, right_bottom_point)
    return crop_rectangles


#  对单张图片进行处理
def split_sample(image_path, output_folder):
    json_filename = image_path.replace(".jpg", ".json")
    raw_image_filename = os.path.basename(image_path).split(".")[0]
    if not os.path.exists(json_filename):
        json_filename = None
    # imgraw = cv2.imread(image_path)
    # cv2.imshow("image", cv2.resize(imgraw,(int(864*1.5), int(1152*1.5))))
    # cv2.waitKey(0)

    # rectangles = generate_strategy_control(json_filename)
    # 截取并保存图像
    # if strategy == "old":
    #     crop_and_save_rectangles(image_path, rectangles, output_folder, raw_image_filename, json_filename)
    if strategy == "16":
        rectangles = get_split_16_rect()
        crop_and_save_rectangles(image_path, rectangles, output_folder, raw_image_filename, json_filename,
                                 resize_factor=1)
    if strategy == "4":
        rectangles = get_split_4_rect()
        crop_and_save_rectangles(image_path, rectangles, output_folder, raw_image_filename, json_filename,
                                 resize_factor=0.5)
    if strategy == "1":
        rectangles = get_split_1_rect()
        crop_and_save_rectangles(image_path, rectangles, output_folder, raw_image_filename, json_filename,
                                 resize_factor=0.25)


#  遍历文件夹，对每个图片进行处理
def split_samples_from_folder(image_folder, output_folder):
    if not os.path.exists(image_folder):
        return
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for image_path in os.listdir(image_folder):
        if image_path.endswith(".jpg"):
            print(image_path)
            split_sample(os.path.join(image_folder, image_path), output_folder)


# main
if __name__ == "__main__":
    image_folder = r""
    output_folder = r""
    split_samples_from_folder(image_folder, output_folder)
    # print(get_split_16_rect())
