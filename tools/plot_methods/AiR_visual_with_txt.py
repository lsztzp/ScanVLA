import argparse
from os.path import join
import json
import numpy as np
import torch
import os 
from pathlib import Path

from tqdm import tqdm
import warnings
import cv2
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.cm as cm
warnings.filterwarnings("ignore")

def plot_scanpath(img_path,scanpaths,save_path="", question="", img_height=320,img_width=512):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    ys = scanpaths[:,0] * height
    xs = scanpaths[:,1] * width

    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    plt.imshow(image)
    plt.axis("off")

    linewidth = 4
    for i in range(len(xs)):
        if i > 0:
            plt.plot([xs[i], xs[i - 1]], [ys[i], ys[i - 1]], color='yellow',linewidth=linewidth, alpha=0.8)

    for i in range(len(xs)):
        # cir_rad = int(14 + rad_per_T * (ts[i] - min_T))
        cir_rad = 15
        circle = plt.Circle((xs[i], ys[i]),
                            radius=cir_rad,
                            facecolor='orange',
                            alpha=0.8)
        ax.add_patch(circle)
        plt.annotate("{}".format(
            i+1), xy=(xs[i], ys[i]+3), fontsize=16, ha="center", va="center")
    
    ax.set_position([0.1, 0.3, 0.8, 0.6])  # 调整图像位置以腾出下方空间

    x = 0.02  # 左侧留一点边距
    y = -0.04
    fontsize=18
    line_height = 0.052
    max_line_width =0.99

    fig = plt.gcf()
    fig.canvas.draw()  # 确保所有尺寸计算基于已渲染的画布

    ax_bbox_pixels = ax.get_window_extent()
    ax_width_pixels = ax_bbox_pixels.width

    for i, word in enumerate(question):
        text_obj = ax.text(
            0, 0,  # 任意位置
            word,
            fontsize=fontsize,
            ha='left',
            va='center',
            transform=ax.transAxes,
        )
    
        # 2. 获取文字在像素坐标系下的宽度
        fig.canvas.draw()
        bbox_pixels = text_obj.get_window_extent(
            renderer=fig.canvas.get_renderer()
        )
        word_width_pixels = bbox_pixels.width
    
        # 3. 计算像素到transAxes的转换比例
        # 因为ax.transAxes的范围是[0,1]，所以：
        pixel_to_ax_ratio = 1.0 / ax_width_pixels
    
        # 4. 计算文字在transAxes中的宽度
        word_width_ax = word_width_pixels * pixel_to_ax_ratio
    
        # 5. 删除临时文字
        text_obj.remove()

        # 换行判断
        if x + word_width_ax > max_line_width:
            x = 0.02
            y -= line_height

        # 正式绘制
        ax.text(
            x, y, word,
            fontsize=fontsize,
            ha='left', va='center',
            transform=ax.transAxes
        )

        x += word_width_ax + 0.014

    ax.axis('off')
    # 保存图像
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)

    plt.close()

if __name__ == '__main__':
    path = 'tools/ploted_images/qualitative_compare_infered_scanpaths/ScanVLA_AiR_infered.pt'
    image_root = '/data/lyt/01-Datasets/01-ScanPath-Datasets/AiR/stimuli/'
    save_root = './tools/ploted_images/qualitative_compare_images/AiR/Ours_with_txt'

    scanpaths = torch.load(path)

    for question_id, performance, idx, scanpath, img_name, question in tqdm(scanpaths):
        image_path = join(image_root, img_name)
        save_path = join(save_root, img_name)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if os.path.exists(save_path):
            continue
        parent_folder = Path(save_path).parent  
        # 父文件夹
        if not parent_folder.is_dir():
            parent_folder.mkdir(parents=True, exist_ok=True)

        question = question.split(' ')
        plot_scanpath(image_path,scanpath,save_path, question,320,512)

    print('done')
