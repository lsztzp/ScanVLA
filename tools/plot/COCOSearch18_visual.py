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
import matplotlib.cm as cm
warnings.filterwarnings("ignore")

# def plot_scanpath(img_path,scanpaths,save_path="",img_height=32*16,img_width=320):
#     image = cv2.resize(matplotlib.image.imread(img_path), (img_width, img_height))

#     plt.imshow(image)

#     points_x = scanpaths[:, 1]
#     points_y = scanpaths[:, 0]

#     colors = cm.rainbow(np.linspace(0, 1, len(points_x)))

#     previous_point = None
#     for num, x, y, c in zip(range(0, len(points_x)), points_x, points_y, colors):
#         markersize = 14.
#         linewidth = 6.
#         if previous_point is not None:
#             if abs(previous_point[0] - x) < (img_width / 2):
#                 plt.plot([x, previous_point[0]], [y, previous_point[1]], color='blue', linewidth=linewidth,
#                          alpha=0.35)
#             else:
#                 h_diff = (y - previous_point[1]) / 2
#                 if x > previous_point[0]:  # X is on the right, Previous is on the Left
#                     plt.plot([previous_point[0], 0],
#                              [previous_point[1], previous_point[1] + h_diff], color='blue',
#                              linewidth=linewidth, alpha=0.35)
#                     plt.plot([img_width, x], [previous_point[1] + h_diff, y],
#                              color='blue', linewidth=linewidth, alpha=0.35)
#                 else:
#                     plt.plot([previous_point[0], img_width],
#                              [previous_point[1], previous_point[1] + h_diff], color='blue',
#                              linewidth=linewidth, alpha=0.35)
#                     plt.plot([0, x], [previous_point[1] + h_diff, y], color='blue', linewidth=linewidth,
#                              alpha=0.35)
#         previous_point = [x, y]
#         plt.plot(x, y, marker='o', markersize=markersize, color=c, alpha=.8)
#     plt.axis('off')
#     plt.margins(0, 0)
#     if not save_path:
#         # plt.show(bbox_inches='tight', pad_inches=-0.1)
#         plt.show()
#     else:
#         plt.savefig(str(save_path), bbox_inches='tight', pad_inches=-0.1, dpi=1000)
#     plt.cla()

def plot_scanpath(img_path,scanpaths,save_path="",img_height=320,img_width=512):
    image = cv2.resize(matplotlib.image.imread(img_path), (img_width, img_height))

    fig, ax = plt.subplots()
    ax.imshow(image)

    ys = scanpaths[:,0]
    xs = scanpaths[:,1]
    ts = scanpaths[:,2]

    cir_rad_min, cir_rad_max = 10,18
    min_T, max_T = np.min(ts), np.max(ts)
    rad_per_T = (cir_rad_max - cir_rad_min) / float(max_T - min_T)

    linewidth = 2
    for i in range(len(xs)):
        if i > 0:
            plt.plot([xs[i], xs[i - 1]], [ys[i], ys[i - 1]], color='red',linewidth=linewidth, alpha=0.35)

    for i in range(len(xs)):
        cir_rad = int(14 + rad_per_T * (ts[i] - min_T))
        circle = plt.Circle((xs[i], ys[i]),
                            radius=cir_rad,
                            facecolor='yellow',
                            alpha=0.5)
        ax.add_patch(circle)
        plt.annotate("{}".format(
            i+1), xy=(xs[i], ys[i]+3), fontsize=10, ha="center", va="center")

    ax.axis('off')
    if not save_path:
        plt.show()
    else:
        parent_dir = os.path.dirname(save_path)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        plt.savefig(str(save_path), bbox_inches='tight', pad_inches=-0.1, dpi=2000)
    plt.cla()


if __name__ == '__main__':
    path = '/data/lyt/03-Repositories/01-ours/ScanVLA/ScanVLA-main/tools/plot/ScanVLA_COCOSearch18_infered.pt'
    image_root = '/data/lyt/01-Datasets/01-ScanPath-Datasets/coco_search18/raw/COCOSearch18/images/'
    save_root = '/data/lyt/03-Repositories/01-ours/ScanVLA/ScanVLA-main/tools/plot/visual_images/COCOSearch18/Ours'

    scanpaths = torch.load(path)

    for task_name, name, condition, idx, scanpath in tqdm(scanpaths):
        image_path = join(image_root, task_name.replace(' ', '_'), name)
        save_path = join(save_root, task_name.replace(' ', '_'), name)

        if os.path.exists(save_path):
            continue
        parent_folder = Path(save_path).parent  
        # 父文件夹
        if not parent_folder.is_dir():
            parent_folder.mkdir(parents=True, exist_ok=True)

        # plot_scanpath(image_path,coordinate,save_path,320,512)
        plot_scanpath(image_path,scanpath,save_path,320,512)


    print('done')
