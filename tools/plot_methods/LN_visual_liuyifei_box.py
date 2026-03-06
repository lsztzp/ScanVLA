# import numpy as np
# from PIL import Image
# import cv2
# import matplotlib.pyplot as plt
# from matplotlib.colors import LinearSegmentedColormap
#
# # 读取.npy格式文件
# # trace_data = np.load('/data/lyf/connect-caption-and-trace-main/data/coco_data/coco_LN_trace_box/6471.npy')
# trace_data = np.load('/data/lyf/connect-caption-and-trace-main/vis/trace_generation_ade20k/pred_trace/ADE_val_00000004.npy')
# # print(trace_data[0])
#
# # 读取图片
# image_path = '/data/lyf/detectron2-main/datasets/ADE20K/images/full_images/ADE_val_00000004.jpg'
# image = Image.open(image_path)
# image = np.array(image)  # 将 PIL 图像转换为 NumPy 数组
#
# # 获取图像的尺寸
# height, width, _ = image.shape
#
# # 创建一个新的图像用于绘制边界框
# fig, ax = plt.subplots()
# ax.imshow(image)
#
# purple_hue = 0.75
# # 使用 'hsv' 颜色映射
# cmap = plt.get_cmap('hsv')
#
# # 绘制每个边界框
# for i, bbox in enumerate(trace_data[0]):
#     x1, y1, x2, y2 = bbox  # 假设最后一列是其他信息，这里忽略
#     # 将归一化的坐标转换为像素坐标
#     x1, x2 = x1 * width, x2 * width
#     y1, y2 = y1 * height, y2 * height
#
#     # 获取颜色映射中的颜色，根据边界框的索引进行归一化
#     color = cmap(i / (len(trace_data[0]) - 1) * purple_hue)
#
#     # 绘制矩形
#     rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor=color, facecolor='none', linewidth=1.5)
#     ax.add_patch(rect)
#
# # 关闭坐标轴
# ax.axis('off')
#
# # 保存图像
# output_path = 'output_image_with_bounding_boxes.png'
# plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
#
# # 显示图像
# plt.show()


import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os

# 定义文件夹路径
# #  ade20k
# npy_folder = '/data/lyf/connect-caption-and-trace-main/vis/trace_generation_ade20k/pred_trace/'
# jpg_folder = '/data/lyf/detectron2-main/datasets/ADE20K/images/full_images/'
# output_folder = '/data/lyf/connect-caption-and-trace-main/visual/trace_generation_ade20k_tvt_gbfm/pred_trace/'

 # coco
npy_folder = '/data/lyf/connect-caption-and-trace-main/vis/trace_generation_coco/pred_trace/'
jpg_folder = '/data/lyf/detectron2-main/datasets/coco/val2014/'
output_folder = '/data/lyf/connect-caption-and-trace-main/visual/trace_generation_coco_gbfmN3/pred_trace/'

# 检查并创建输出文件夹
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 获取所有.npy文件
npy_files = sorted([f for f in os.listdir(npy_folder) if f.endswith('.npy')])

# 遍历所有.npy文件
for npy_file in npy_files:
    # 读取.npy格式文件
    trace_data = np.load(os.path.join(npy_folder, npy_file))

    #coco
    #----------------------------------------------
    # 提取文件名和后缀
    filename, extension = npy_file.split('.')
    # 使用 zfill 方法扩充数字部分
    new_filename = filename.zfill(12)
    # 拼接新的字符串
    npy_file = 'COCO_val2014_' + new_filename + '.' + extension
    # ----------------------------------------------

    # 获取对应的.jpg图片文件名
    jpg_file = npy_file.replace('.npy', '.jpg')
    image_path = os.path.join(jpg_folder, jpg_file)
    image = Image.open(image_path)
    image = np.array(image)  # 将 PIL 图像转换为 NumPy 数组

    # 检查图像是否为灰度图像
    if len(image.shape) == 2:
        # 将灰度图像转换为三通道图像
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # 获取图像的尺寸
    height, width, _ = image.shape

    # 创建一个新的图像用于绘制边界框
    fig, ax = plt.subplots()
    ax.imshow(image)

    purple_hue = 0.75
    # 使用 'hsv' 颜色映射
    cmap = plt.get_cmap('hsv')

    # 绘制每个边界框
    for i, bbox in enumerate(trace_data[0]):
        x1, y1, x2, y2 = bbox  # 假设最后一列是其他信息，这里忽略
        # 将归一化的坐标转换为像素坐标
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height

        # 获取颜色映射中的颜色，根据边界框的索引进行归一化
        color = cmap(i / (len(trace_data[0]) - 1) * purple_hue)

        # 绘制矩形
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor=color, facecolor='none', linewidth=1.5)
        ax.add_patch(rect)

    # 关闭坐标轴
    ax.axis('off')

    # 保存图像
    output_path = os.path.join(output_folder, 'output_' + jpg_file)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)

    # 显示图像
    # plt.show()

    # 关闭图形，释放内存
    plt.close()
