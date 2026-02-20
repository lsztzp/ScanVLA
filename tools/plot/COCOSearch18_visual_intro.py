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
from PIL import Image

import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def plot_scanpath(img_path,scanpaths,save_path="",img_height=320,img_width=512, text=None):
    image = Image.open(img_path).convert("RGB")
    width, height = image.size

    # image = image.resize((img_width, img_height), resample=Image.Resampling.LANCZOS)
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    plt.imshow(image)
    plt.axis("off")

    ys = scanpaths[0, :]   * height
    xs = scanpaths[1, :]   * width

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

    ax.axis('off')
    parent_dir = os.path.dirname(save_path)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()


if __name__ == '__main__':
    path = '/data/lyt/03-Repositories/01-ours/ScanVLA/ScanVLA-main/tools/plot/intro/cocosearch18.pt'
    image_root = '/data/lyt/01-Datasets/01-ScanPath-Datasets/coco_search18/raw/COCOSearch18/images/'
    save_root = '/data/lyt/03-Repositories/01-ours/ScanVLA/ScanVLA-main/tools/plot/intro/visual'

    scanpaths = torch.load(path)

    cnt = 0
    for elem in tqdm(scanpaths):
        elem['tgt_seq_y'] = elem['tgt_seq_y'] /320
        elem['tgt_seq_x'] = elem['tgt_seq_x'] /512        

        task_name = elem['task']
        name = elem['img_name']
        # image_path = join(image_root, task_name.replace(' ', '_'), name)
        image_path = "/data/lyt/03-Repositories/01-ours/ScanVLA/ScanVLA-main/tools/plot/intro/45775.jpg"
        save_path = join(save_root, task_name.replace(' ', '_'),  name)
        scanpath = torch.stack([elem['tgt_seq_y'],elem['tgt_seq_x']], dim=0)
        
        # X = [247, 175, 162, 164] 
        # Y = [152, 164, 178, 182] 

        # scanpath = torch.tensor([[0.4546, 0.4963, 0.5074, 0.93, 0.8461, 0.5145],
        #                                 [0.4943, 0.4721, 0.6271, 0.81, 0.2620, 0.1510]])
        # scanpath = torch.tensor([Y,X])
        scanpath = torch.tensor([[0.4769, 0.52, 0.44],
                                        [0.4763, 0.225, 0.18]])
        plot_scanpath(image_path,scanpath,save_path,320,512)
        cnt += 1

    print('done')
