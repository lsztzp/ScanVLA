import os
import numpy as np
import warnings
import cv2
import torch
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

def plot_scanpath(img_path, xs, ys, save_path="",img_height=320,img_width=512):
    image = Image.open(img_path).convert("RGB")
    image = image.resize((img_width, img_height), resample=Image.Resampling.LANCZOS)

    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    plt.imshow(image)
    plt.axis("off")

    linewidth = 2
    # colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FFA500', '#FF00FF', '#FFC0CB', '#808080', '#5C3317', '#B0C4DE', '#87CEEB', '#FFB6C1', '#000000']
    
    colors = ['red', 'Green', 'Blue', 'Purple', 'Orange', 'Gray', 'pink', 'Olive', 'aqua', 'Navy', 'OrangeRed', 'Crimson', 'Magenta', 'SlateBlue', 'Gold' ]
    # colors = ['#FF0000', '#00FFFF', '#FFC0CB', '#008000', '#0000FF', '#800080', '#FFA500', '#808080', '#808000', '#000080', '#FF4500', '#DC143C', '#FF00FF', '#6A5ACD', '#FFD700']

    # x, y, width, height = bbox
    # x1, y1 = x, y
    # x2, y2 = x + width, y
    # x3, y3 = x + width, y + height
    # x4, y4 = x, y + height

    cnt = 1
    last_x, last_y = 256, 160
    for i in range(len(xs)):
        color = colors[i]

        for j in range(len(xs[i])):
            if j > 0:
                # plt.plot([xs[i], xs[i - 1]], [ys[i], ys[i - 1]], color='red',linewidth=linewidth, alpha=0.35)
                plt.plot([xs[i][j], xs[i][j-1]], [ys[i][j], ys[i][j-1]], color='yellow',linewidth=linewidth, alpha=0.4)

        if i > 0 and xs[i]:
            plt.plot([xs[i][0], last_x], [ys[i][0], last_y], color='yellow',linewidth=linewidth, alpha=0.4)

        for j in range(len(xs[i])):
            # cir_rad = int(14 + rad_per_T * (ts[i] - min_T))
            cir_rad = 14
            circle = plt.Circle((xs[i][j], ys[i][j]),
                            radius=cir_rad,
                            facecolor=color,
                            alpha=0.75,
                            edgecolor='grey',     # 边框色（默认黑色，可自定义）
                            linewidth=1,         # 边框宽度（单位：点，1pt≈0.35mm）
                            )
            ax.add_patch(circle)
            plt.annotate("{}".format(
                cnt), xy=(xs[i][j], ys[i][j]), fontsize=10, ha="center", va="center")
            last_x, last_y = xs[i][j], ys[i][j]
            cnt += 1

    ax.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()


if __name__=='__main__':
    # path = 'tools/plot/ScanVLA_RefCOCOGaze_infered.pt'
    path = '/data/lyt/03-Repositories/01-ours/ScanVLA/ScanVLA-main/tools/plot/intro/refcoco_gaze.pt'
    image_root = '/data/lyt/01-Datasets/01-ScanPath-Datasets/ART_data/data/images_512X320/'
    save_root = '/data/lyt/03-Repositories/01-ours/ScanVLA/ScanVLA-main/tools/plot/intro/refcocogaze'

    scanpaths = torch.load(path)

    # a = []
    cnt = 0
    for elem in scanpaths:
        # image_path = elem['IMAGEFILE']
        # image_path = image_root + elem[1]
        image_path = "/data/lyt/03-Repositories/01-ours/ScanVLA/ScanVLA-main/tools/plot/intro/45775.jpg"
        save_path = os.path.join(save_root, elem[1])

        X = torch.tensor(elem[4][:,1]).unsqueeze(dim=1)
        Y = torch.tensor(elem[4][:,0]).unsqueeze(dim=1)
        # X,Y = elem['X'],elem['Y']
        # bbox = elem['BBOX']
        # text = elem['TEXT']

        # if os.path.exists(save_path):
        #     continue
        cnt+=1

        # [[152.0, 247.0, 233.0], [164.0, 175.0, 165.0], [178.0, 162.0, 229.0], [182.0, 164.0, 241]

        # text = text[:-18] + ' [EOT]'
        # text = text.split(' ')
        # text[0], text[-1] = '[BOT]', '[EOT]'
        # X = [[247], [175], [162], [164]]
        # Y = [[152], [164], [178], [182]]
        # X = [[255.0], [235.0], [266.0], [368.0, 372], [394.0]]
        # Y = [[159.0], [151.0], [125.5], [92.0, 90], [74]]

        X = [[255.0], [235.0], [256.0, 275], [342.0], [362.0, 372], [410.0]]
        Y = [[159.0], [151.0], [119.5, 108], [102.0], [98.0, 92], [79.5]]
        # X = [[255.0], [231.0], [256.0], [280.0], [394.0]]
        # Y = [[159.0], [169.0], [120.5], [119.5], [93.5]]
        plot_scanpath(image_path, xs= X, ys = Y, save_path=save_path)
        print(image_path)
    # save_path = "/data/lyt/03-Repositories/01-ours/ScanVLA/ScanVLA-main/tools/imageid/RefCOCOGaze_ids.pt"
    # torch.save(a,save_path)

    print('done')

