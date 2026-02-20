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
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
warnings.filterwarnings("ignore")

def plot_scanpath(img_path,scanpaths,save_path="",img_height=320,img_width=512, text=None):
    image = Image.open(img_path).convert("RGB")
    image = image.resize((img_width, img_height), resample=Image.Resampling.LANCZOS)
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    plt.imshow(image)
    plt.axis("off")
    # image = cv2.resize(matplotlib.image.imread(img_path), (img_width, img_height))

    # fig, ax = plt.subplots()
    # ax.imshow(image)

    ys = scanpaths[:,0]
    xs = scanpaths[:,1]
    # ts = scanpaths[:,2]

    # cir_rad_min, cir_rad_max = 10,18
    # min_T, max_T = np.min(ts), np.max(ts)
    # rad_per_T = (cir_rad_max - cir_rad_min) / float(max_T - min_T)

    linewidth = 2
    for i in range(len(xs)):
        if i > 0:
            plt.plot([xs[i], xs[i - 1]], [ys[i], ys[i - 1]], color='red',linewidth=linewidth, alpha=0.35)

    for i in range(len(xs)):
        # cir_rad = int(14 + rad_per_T * (ts[i] - min_T))
        cir_rad = 10
        circle = plt.Circle((xs[i], ys[i]),
                            radius=cir_rad,
                            facecolor='yellow',
                            alpha=0.5)
        ax.add_patch(circle)
        plt.annotate("{}".format(
            i+1), xy=(xs[i], ys[i]+3), fontsize=10, ha="center", va="center")

    # 在图像下方标注单词
    # num_words = len(text)
    # colors = plt.cm.viridis(np.linspace(0, 1, num_words))
    

    ax.set_position([0.1, 0.3, 0.8, 0.6])  # 调整图像位置以腾出下方空间

    x = 0.02  # 左侧留一点边距
    y = -0.02
    fontsize=12
    line_height = 0.04
    max_line_width =0.99

    fig = plt.gcf()
    fig.canvas.draw()  # 确保所有尺寸计算基于已渲染的画布

    ax_bbox_pixels = ax.get_window_extent()
    ax_width_pixels = ax_bbox_pixels.width

    for i, word in enumerate(text):
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
            # color=colors[i],
            ha='left', va='center',
            transform=ax.transAxes
        )

        x += word_width_ax + 0.01

    ax.axis('off')
    parent_dir = os.path.dirname(save_path)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()



if __name__ == '__main__':
    path = '/data/lyt/03-Repositories/01-ours/ScanVLA/ScanVLA-main/tools/plot/ScanVLA_COCOSearch18_infered.pt'
    image_root = '/data/lyt/01-Datasets/01-ScanPath-Datasets/coco_search18/raw/COCOSearch18/images/'
    save_root = '/data/lyt/03-Repositories/01-ours/ScanVLA/ScanVLA-main/tools/plot/visual_images/COCOSearch18/Ours_with_txt'

    scanpaths = torch.load(path)

    # arr = []
    for task_name, name, condition, idx, scanpath in tqdm(scanpaths):
        image_id = int(name.split('.')[0])
        # arr.append(image_id)

        image_path = join(image_root, task_name.replace(' ', '_'), name)
        save_path = join(save_root, task_name.replace(' ', '_'), name)

        if os.path.exists(save_path):
            continue
        parent_folder = Path(save_path).parent  
        # 父文件夹
        if not parent_folder.is_dir():
            parent_folder.mkdir(parents=True, exist_ok=True)

        # plot_scanpath(image_path,coordinate,save_path,320,512)
        text = [task_name]
        plot_scanpath(image_path,scanpath,save_path,320,512, text)

    # save_path = "/data/lyt/03-Repositories/01-ours/ScanVLA/ScanVLA-main/tools/imageid/COCOSearch18_ids.pt"
    # torch.save(arr,save_path)

    print('done')
