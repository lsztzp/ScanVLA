import torch
import cv2
import numpy as np
import skimage as sk
import matplotlib.pyplot as plt
import os


# image_path="/data/qmengyu/01-Datasets/01-ScanPath-Dataset/sitzmann/images/cubemap_0009.png"
# scanpath_path="/data/qmengyu/01-Datasets/01-ScanPath-Dataset/sitzmann/fixations/cubemap_0009.pck"
# scanpath_path="/data/qmengyu/02-Results/01-ScanPath/360_Compare_results/plot_ans/cubemap_0009_50paths.pck"
# scanpath_path="/data/qmengyu/02-Results/01-ScanPath/360_Compare_results/time-saliency/cubemap_0000_1000paths.pck"
# scanpaths=torch.load(scanpath_path)


def sphere2plane(sphere_cord, height_width=None):
    """ input:  (lat, lon) shape = (n, 2)
        output: (x, y) shape = (n, 2) """
    lat, lon = sphere_cord[:, 0], sphere_cord[:, 1]
    if height_width is None:
        y = (lat + 90) / 180
        x = (lon + 180) / 360
    else:
        y = (lat + 90) / 180 * height_width[0]
        x = (lon + 180) / 360 * height_width[1]
    return torch.cat((y.view(-1, 1), x.view(-1, 1)), 1)


def create_saliency_map(image_path, keypoints, size=(512, 1024)):
    # 读取图像
    # image = cv2.imread(image_path)
    # image =cv2.resize(image,(1024,512))

    # 创建空白的显著图，全黑
    saliency_map = np.zeros(size)

    # 在显著图上绘制这些点，使其值更高
    for point in keypoints:
        y, x = point[0], point[1]
        y, x = min(y, size[0] - 1), min(x, size[1] - 1)
        saliency_map[y, x] = 255  # 像素值更高，表示更显著

    # 使用高斯模糊使显著图更平滑（可选）
    saliency_map = cv2.GaussianBlur(saliency_map, (191, 191), 0)

    # # 将显著图转换为RGB图像格式
    # saliency_map = cv2.merge([saliency_map, saliency_map, saliency_map])
    #
    # # 将显著图与原始图像叠加以突出显著区域
    # blended_image = cv2.addWeighted(image, 0.7, saliency_map, 0.3, 0)

    return saliency_map


# if __name__=="__main__":
#     # resize=(512,1024)
#
#     image_path = "/data/qmengyu/01-Datasets/01-ScanPath-Dataset/salient360/images/"
#     scanpath_path = "/data/qmengyu/01-Datasets/01-ScanPath-Dataset/salient360/fixations/"
#     output_path="/data/qmengyu/02-Results/01-ScanPath/360_Compare_results/plot_ans/time-saliency/salient360/"
#     pck_path="/data/qmengyu/02-Results/01-ScanPath/360_Compare_results/plot_ans/pck_file/"
#
#     image_dir=os.listdir(output_path)
#     for file in image_dir:
#         image_path_one = image_path + file + '.png'
#         image = cv2.imread(image_path_one)
#
#         scanpath_path_one = pck_path + file + '_50paths.pck'
#         old_scanpaths=scanpath_path+file+'.pck'
#         old_scanpaths=torch.load(old_scanpaths)
#         n=len(old_scanpaths)
#         scanpaths = torch.load(scanpath_path_one)[:min(n,50)]
#
#         save_path =output_path + file+'/ours/'
#
#         if not os.path.exists(save_path):
#             os.makedirs(save_path)
#         # scanpaths = torch.load(scanpath_path)[:36]
#
#         image = cv2.resize(image, (1024, 512))
#         resize = image.shape[:2]
#         for i in range(len(scanpaths)):
#             scanpaths[i] = sphere2plane(scanpaths[i], resize)
#
#         scanpaths = scanpaths.int()
#         # save_path="/data/qmengyu/02-Results/01-ScanPath/360_Compare_results/plot_ans/time-saliency/cubemap0009/ours/"
#         for i in range(6):
#             points = scanpaths[:, 5 * i:5 * (i + 1), :]
#             points = points.contiguous().view(-1, 2)
#             saliency_map = create_saliency_map(image_path, points, size=resize)
#
#             plt.imshow(image)
#             plt.imshow(saliency_map, cmap='jet', alpha=0.6)
#             plt.axis("off")
#             plt.xticks([])
#             plt.yticks([])
#             plt.savefig(save_path + str(i) + '.png', bbox_inches='tight', pad_inches=-0.1, dpi=600)
#             plt.show()
#
#     print('done')


if __name__ == "__main__":
    image_path = "/data/qmengyu/01-Datasets/01-ScanPath-Dataset/JUFE/images/"
    scanpath_path = "/data/qmengyu/01-Datasets/01-ScanPath-Dataset/JUFE/fixations/"
    output_path = "/data/qmengyu/02-Results/01-ScanPath/360_Compare_results/plot_ans/time-saliency/jufe/"

    cnt = 0
    for file in os.listdir(image_path):
        # if (file[-6] == "_"):
        #     continue

        cnt += 1
        if cnt >= 10:
            print('done!')
            exit()

        name = file.split('.')[0]
        save_path = output_path + name + '/gt/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        scanpath_path_one = scanpath_path + name + '.pck'
        scanpaths = torch.load(scanpath_path_one)

        # scanpaths = torch.load(scanpath_path)[:36]
        image_path_one = image_path + file
        image = cv2.imread(image_path_one)

        image = cv2.resize(image, (1024, 512))
        resize = image.shape[:2]
        for i in range(len(scanpaths)):
            scanpaths[i] = sphere2plane(scanpaths[i], resize)

        scanpaths = scanpaths.int()
        # save_path="/data/qmengyu/02-Results/01-ScanPath/360_Compare_results/plot_ans/time-saliency/cubemap0009/ours/"
        for i in range(3):
            points = scanpaths[:, 5 * i:5 * (i + 1), :]
            points = points.contiguous().view(-1, 2)
            saliency_map = create_saliency_map(image_path, points, size=resize)

            plt.imshow(image)
            plt.imshow(saliency_map, cmap='jet', alpha=0.6)
            plt.axis("off")
            plt.xticks([])
            plt.yticks([])
            plt.savefig(save_path + str(i) + '.png', bbox_inches='tight', pad_inches=-0.1, dpi=600)
            plt.show()

    print('done!')
