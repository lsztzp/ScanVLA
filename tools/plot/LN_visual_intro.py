import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
import torch
from tqdm import tqdm
from os.path import join
from matplotlib.text import TextPath
from scipy.interpolate import make_interp_spline


def plot_scanpath_color(image_path, save_path, center_x, center_y, caption_utterance):
    # 读取图像
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    # 将归一化坐标映射到图像像素坐标
    center_x_pixels = np.array((center_x.flatten() * width)).astype(int)
    center_y_pixels = np.array((center_y.flatten() * height)).astype(int)

    # 创建渐变颜色
    num_points = len(center_x)

    colors = plt.cm.viridis(np.linspace(0, 1, num_points))

    # 绘制图像
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    plt.imshow(image)
    plt.axis("off")
    
    # 平滑处理
    if num_points >= 2:
        # 参数化t值，用于插值
        t = np.linspace(0, 1, num_points)
        # 生成更密集的t值，使曲线更平滑（这里是10倍密度）
        t_smooth = np.linspace(0, 1, num_points * 10)
        
        # 根据点数选择插值方式，避免报错
        k = 3 if num_points >= 4 else 1  # 点数>=4用三次样条，否则用线性插值
        spline_x = make_interp_spline(t, center_x_pixels, k=k)
        spline_y = make_interp_spline(t, center_y_pixels, k=k)
        
        # 计算平滑后的坐标
        x_smooth = spline_x(t_smooth)
        y_smooth = spline_y(t_smooth)

        # 绘制平滑轨迹（保持颜色渐变）
        for i in range(len(t_smooth) - 1):
            # 根据当前点在原始序列中的比例，获取对应颜色
            color_ratio = t_smooth[i]
            line_color = plt.cm.viridis(color_ratio)
            ax.plot(
                [x_smooth[i], x_smooth[i+1]],
                [y_smooth[i], y_smooth[i+1]],
                color=line_color,
                linewidth=2,
                alpha=0.8
            )

    num_words = len(caption_utterance)
    colors = plt.cm.viridis(np.linspace(0, 1, num_words))

    # # 在图像下方标注单词
    ax.set_position([0.1, 0.3, 0.8, 0.6])  # 调整图像位置以腾出下方空间

    # x = 0.02  # 左侧留一点边距
    # y = -0.02
    # fontsize=12
    # line_height = 0.04
    # max_line_width =0.99

    # fig = plt.gcf()
    # fig.canvas.draw()  # 确保所有尺寸计算基于已渲染的画布

    # ax_bbox_pixels = ax.get_window_extent()
    # ax_width_pixels = ax_bbox_pixels.width

    # for i, word in enumerate(caption_utterance):
    #     text_obj = ax.text(
    #         0, 0,  # 任意位置
    #         word,
    #         fontsize=fontsize,
    #         ha='left',
    #         va='center',
    #         transform=ax.transAxes,
    #     )
    
    #     # 2. 获取文字在像素坐标系下的宽度
    #     fig.canvas.draw()
    #     bbox_pixels = text_obj.get_window_extent(
    #         renderer=fig.canvas.get_renderer()
    #     )
    #     word_width_pixels = bbox_pixels.width
    
    #     # 3. 计算像素到transAxes的转换比例
    #     # 因为ax.transAxes的范围是[0,1]，所以：
    #     pixel_to_ax_ratio = 1.0 / ax_width_pixels
    
    #     # 4. 计算文字在transAxes中的宽度
    #     word_width_ax = word_width_pixels * pixel_to_ax_ratio
    
    #     # 5. 删除临时文字
    #     text_obj.remove()

    #     # 换行判断
    #     if x + word_width_ax > max_line_width:
    #         x = 0.02
    #         y -= line_height

    #     # 正式绘制
    #     ax.text(
    #         x, y, word,
    #         fontsize=fontsize,
    #         color=colors[i],
    #         ha='left', va='center',
    #         transform=ax.transAxes
    #     )

    #     x += word_width_ax + 0.01

    ax.axis('off')
    # 保存图像
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)

    plt.close()


if __name__=="__main__":
    # path = '/data/lyt/03-Repositories/01-ours/ScanVLA/ScanVLA-main/tools/plot/ScanVLA_LN_infered.pt'
    path = '/data/lyt/03-Repositories/01-ours/ScanVLA/ScanVLA-main/tools/plot/intro/LN.pt'
    image_root = '/data/lyt/01-Datasets/01-ScanPath-Datasets/coco/val2017/'
    save_root = './tools/plot/visual_images/LN/Ours'

    scanpaths = torch.load(path)

    # 检查并创建输出文件夹
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    cnt = 0
    for elem in tqdm(scanpaths):
        # caption = elem['caption'].split(' ')
        caption = "A breakfast table with a sandwich on a plate, sliced fruit, one fork and one knife on a napkin, two yellow mugs, and a clear glass cup in the background".split(' ')
        # IMG_ID = elem['IMG_ID']
        # predict_bbox = elem['predict_bbox'].cpu().to(torch.float32)
        predict_center_x = elem['predict_center_x'].cpu().to(torch.float32)
        predict_center_y = elem['predict_center_y'].cpu().to(torch.float32)
        
        predict_center_x = torch.tensor([[0.4180, 0.4863, 0.4824, 0.4961, 0.4902, 0.4395, 0.4629, 0.4824, 0.4980,
            0.5273, 0.5469, 0.5586, 0.5859, 0.7070, 0.2041, 0.2090, 0.1934, 0.2061,
            0.2129, 0.1904, 0.1494, 0.1426, 0.1953, 0.6719, 0.7812, 0.7812, 0.7812,
            0.7461, 0.6602, 0.5898, 0.7227, 0.7227, 0.73, 0.7892, 0.8506, 0.7477]])
        predict_center_y = torch.tensor([[0.4648, 0.5391, 0.5234, 0.5586, 0.6094, 0.5000, 0.5273, 0.5508, 0.5430,
         0.5352, 0.4863, 0.4609, 0.4668, 0.4180, 0.5000, 0.5352, 0.5625, 0.5703,
         0.5938, 0.5898, 0.5391, 0.5391, 0.5430, 0.4199, 0.4023, 0.3926, 0.4023,
         0.3457, 0.3340, 0.3242, 0.1797, 0.1504, 0.1402, 0.1412, 0.1030, 0.0827]])

        # predict_center_x = elem['predict_center_x'].cpu().to(torch.float32)
        # predict_center_y = elem['predict_center_x'].cpu().to(torch.float32)

        # new_filename = IMG_ID.zfill(12)

        # save_path = join(save_root, new_filename + '_' +str(cnt) + '.jpg')
        save_path = "/data/lyt/03-Repositories/01-ours/ScanVLA/ScanVLA-main/tools/plot/intro/LN/45775.jpg"

        cnt += 1

        # if os.path.exists(save_path):
        #     continue

        # image_path = join(image_root, new_filename + '.jpg')   
        image_path = "/data/lyt/03-Repositories/01-ours/ScanVLA/ScanVLA-main/tools/plot/intro/45775.jpg"

        plot_scanpath_color(image_path, save_path, predict_center_x[0], predict_center_y[0], caption)
        print('done')
    print('done')

