import os
import numpy as np
import warnings
import cv2
import torch
import matplotlib
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

def plot_scanpath(img_path, xs, ys, save_path="",img_height=320,img_width=512, bbox=None, text =None):
    image = cv2.resize(matplotlib.image.imread(img_path), (img_width, img_height))

    fig, ax = plt.subplots()
    ax.imshow(image)

    linewidth = 2
    # colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FFA500', '#FF00FF', '#FFC0CB', '#808080', '#5C3317', '#B0C4DE', '#87CEEB', '#FFB6C1', '#000000']
    
    colors = ['red', 'aqua', 'pink', 'Green', 'Blue', 'Purple', 'Orange', 'Gray', 'Olive', 'Navy', 'OrangeRed', 'Crimson', 'Magenta', 'SlateBlue', 'Gold' ]
    # colors = ['#FF0000', '#00FFFF', '#FFC0CB', '#008000', '#0000FF', '#800080', '#FFA500', '#808080', '#808000', '#000080', '#FF4500', '#DC143C', '#FF00FF', '#6A5ACD', '#FFD700']

    x, y, width, height = bbox
    x1, y1 = x, y
    x2, y2 = x + width, y
    x3, y3 = x + width, y + height
    x4, y4 = x, y + height

    # 绘制矩形
    plt.plot([x1, x2, x3, x4, x1], [y1, y2, y3, y4, y1], 'b-', linewidth=1)

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
    if not save_path:
        plt.show(bbox_inches='tight', pad_inches=-0.1)
    else:
        # parent_dir = os.path.dirname(save_path)
        # if not os.path.exists(parent_dir):
        #     os.makedirs(parent_dir)
        plt.savefig(str(save_path), bbox_inches='tight', pad_inches=-0.1, dpi=2000)
    plt.cla()


if __name__=='__main__':
    # x = [[255.01210021972656], [232.01780700683594], [318.61968994140625], [389.07366943359375, 395.53857421875]]
    # y = [[159.66329956054688], [162.040771484375], [135.76873779296875], [110.5812759399414, 105.73650360107422]]
    # image_path = '' + '34037.jpg'
    # text = '<|object_ref_start|> silver benz <|object_ref_end|>'
    # bbox = [348, 71, 120, 86]
    # # image_root = '/data/lyt/01-Datasets/01-ScanPath-Datasets/coco_search18/raw/COCOSearch18/images/car/'
    # image_root = '/data/lyt/01-Datasets/01-ScanPath-Datasets/ART_data/data/images_512X320/'
    # plot_scanpath(image_root + image_path, xs= x, ys = y, bbox=bbox, save_path='marked_image.jpg', text=text)

    # path = '/data/lyt/02-Results/01-ScanPath/ScanLLM/Qualititive/ours.pt'
    # image_root = '/data/lyt/01-Datasets/01-ScanPath-Datasets/ART_data/data/images_512X320/'
    # save_root = '/data/lyt/02-Results/01-ScanPath/ScanLLM/Qualititive/Our_new/'

    # path = '/data/lyt/02-Results/01-ScanPath/ScanLLM/Qualititive/art.pt'
    # image_root = '/data/lyt/01-Datasets/01-ScanPath-Datasets/ART_data/data/images_512X320/'
    # save_root = '/data/lyt/02-Results/01-ScanPath/ScanLLM/Qualititive/ART_new/'

    # path = '/data/lyt/02-Results/01-ScanPath/ScanLLM/Qualititive/huaman.pt'
    # image_root = '/data/lyt/01-Datasets/01-ScanPath-Datasets/ART_data/data/images_512X320/'
    # save_root = '/data/lyt/02-Results/01-ScanPath/ScanLLM/Qualititive/Human_new/'

    path = 'tools/plot/ScanVLA_RefCOCOGaze_infered.pt'
    image_root = '/data/lyt/01-Datasets/01-ScanPath-Datasets/ART_data/data/images_512X320/'
    save_root = './tools/plot/visual_images/RefCOCOGaze/Ours'

    scanpaths = torch.load(path)

    cnt = 0
    for elem in scanpaths:
        image_path = elem['IMAGEFILE']
        X,Y = elem['X'],elem['Y']
        bbox = elem['BBOX']
        text = elem['TEXT']

        # save_path = save_root + image_path.split('.')[0] + '_' + str(cnt) +'.jpg'
        save_path = os.path.join(save_root, image_path)
        if os.path.exists(save_path):
            continue
        cnt+=1

        plot_scanpath(image_root + image_path, xs= X, ys = Y, bbox=bbox, save_path=save_path, text=text)
        print(image_path)

    print('done')

