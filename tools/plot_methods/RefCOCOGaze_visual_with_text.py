import os
import numpy as np
import warnings
import cv2
import torch
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

def plot_scanpath(img_path, xs, ys, save_path="",img_height=320,img_width=512, bbox=None, text =None):
    image = Image.open(img_path).convert("RGB")
    image = image.resize((img_width, img_height), resample=Image.Resampling.LANCZOS)

    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    plt.imshow(image)
    plt.axis("off")

    linewidth = 4 #轨迹的粗细

    colors = ['red', 'Purple', 'Green', 'Blue', 'Orange', 'Gray', 'pink', 'Olive', 'aqua', 'Navy', 'OrangeRed', 'Crimson', 'Magenta', 'SlateBlue', 'Gold' ]
    # colors = ['#FF0000', '#00FFFF', '#FFC0CB', '#008000', '#0000FF', '#800080', '#FFA500', '#808080', '#808000', '#000080', '#FF4500', '#DC143C', '#FF00FF', '#6A5ACD', '#FFD700']

    x, y, width, height = bbox
    x1, y1 = x, y
    x2, y2 = x + width, y
    x3, y3 = x + width, y + height
    x4, y4 = x, y + height

    # 绘制矩形
    # plt.plot([x1, x2, x3, x4, x1], [y1, y2, y3, y4, y1], 'b-', linewidth=1)

    cnt = 1
    last_x, last_y = 256, 160
    for i in range(len(xs)):
        color = colors[i]

        for j in range(len(xs[i])):
            if j > 0:
                # plt.plot([xs[i], xs[i - 1]], [ys[i], ys[i - 1]], color='red',linewidth=linewidth, alpha=0.35)
                plt.plot([xs[i][j], xs[i][j-1]], [ys[i][j], ys[i][j-1]], color='yellow',linewidth=linewidth, alpha=0.8)

        if i > 0 and xs[i]:
            plt.plot([xs[i][0], last_x], [ys[i][0], last_y], color='yellow',linewidth=linewidth, alpha=0.8)

        for j in range(len(xs[i])):
            cir_rad = 15
            circle = plt.Circle((xs[i][j], ys[i][j]),
                            radius=cir_rad,
                            facecolor=color,
                            alpha=0.8,
                            edgecolor='grey',     # 边框色（默认黑色，可自定义）
                            linewidth=1,         # 边框宽度（单位：点，1pt≈0.35mm）
                            )
            ax.add_patch(circle)
            plt.annotate("{}".format(
                cnt), xy=(xs[i][j], ys[i][j]), fontsize=16, ha="center", va="center")
            last_x, last_y = xs[i][j], ys[i][j]
            cnt += 1


    # 在图像下方标注单词
    ax.set_position([0.1, 0.3, 0.8, 0.6])  # 调整图像位置以腾出下方空间

    x = 0.02  # 左侧留一点边距
    y = -0.05
    fontsize=25
    line_height = 0.08
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
            color=colors[i],
            ha='left', va='center',
            transform=ax.transAxes
        )

        x += word_width_ax + 0.02

    ax.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()


if __name__=='__main__':
    image_root = '/data/lyt/01-Datasets/01-ScanPath-Datasets/ART_data/data/images_512X320/'
    # path = 'tools/ploted_images/qualitative_compare_infered_scanpaths/ScanVLA_RefCOCOGaze_infered.pt'
    # save_root = './tools/ploted_images/qualitative_compare_images/RefCOCOGaze/Ours_with_text'

    # path = 'tools/ploted_images/qualitative_compare_infered_scanpaths/Others/RefCOCOGaze_ART_infered.pt'
    # save_root = 'tools/ploted_images/qualitative_compare_images/RefCOCOGaze/ART_with_text'

    path = 'tools/ploted_images/qualitative_compare_infered_scanpaths/Others/RefCOCOGaze_Human_infered.pt'
    save_root = 'tools/ploted_images/qualitative_compare_images/RefCOCOGaze/Human_with_text'

    scanpaths = torch.load(path)

    # a = []
    cnt = 0
    for elem in scanpaths:
        image_path = elem['IMAGEFILE']

        X,Y = elem['X'],elem['Y']
        bbox = elem['BBOX']
        text = elem['TEXT']

        save_path = os.path.join(save_root, str(cnt) + '_' + image_path)
        cnt +=1 
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        if os.path.exists(save_path):
            continue

        # 针对我们的pth文件，打开以下代码
        # text = text[:-18] + ' [EOT]'
        # text = text.split(' ')
        # text[0], text[-1] = '[BOT]', '[EOT]'

        text = text.split(' ')
        text = ['[BOT]'] + text + ['[EOT]']
        plot_scanpath(image_root + image_path, xs= X, ys = Y, bbox=bbox, save_path=save_path, text=text)
        print(image_path)

    print('done')
