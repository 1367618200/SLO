import argparse
import os

import cv2
import numpy as np

extensions = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif')


# a standard function for reference
def mean_std(root, print_interval=100):
    sum_mean = np.zeros(3)
    sum_std = np.zeros(3)

    num = 0
    for root, dirs, files in os.walk(root):
        for file in files:
            if file.endswith(extensions):
                img = cv2.imread(os.path.join(root, file))
                mean_b = np.mean(img[:, :, 0])
                mean_g = np.mean(img[:, :, 1])
                mean_r = np.mean(img[:, :, 2])
                std_b = np.std(img[:, :, 0])
                std_g = np.std(img[:, :, 1])
                std_r = np.std(img[:, :, 2])

                mean_rgb = np.array([mean_r, mean_g, mean_b])
                std_rgb = np.array([std_r, std_g, std_b])
                sum_mean = sum_mean + mean_rgb
                sum_std = sum_std + std_rgb

                num += 1
                if print_interval > 0 and num % print_interval == 0:
                    print(f'[{num}]', os.path.join(root, file))
    mean_rgb = sum_mean / (1. * num)
    std_rgb = sum_std / (1. * num)
    return mean_rgb, std_rgb, num


# for GAMMA
def mean_std_fundus(root, print_interval=100):
    sum_mean = np.zeros(3)
    sum_std = np.zeros(3)

    num = 0
    for dir in os.listdir(root):
        for file in os.listdir(os.path.join(root, dir)):
            if file.endswith(extensions):
                file_path = os.path.join(root, dir, file)
                img = cv2.imread(file_path)
                mean_b = np.mean(img[:, :, 0])
                mean_g = np.mean(img[:, :, 1])
                mean_r = np.mean(img[:, :, 2])
                std_b = np.std(img[:, :, 0])
                std_g = np.std(img[:, :, 1])
                std_r = np.std(img[:, :, 2])

                mean_rgb = np.array([mean_r, mean_g, mean_b])
                std_rgb = np.array([std_r, std_g, std_b])
                sum_mean = sum_mean + mean_rgb
                sum_std = sum_std + std_rgb

                num += 1
                if print_interval > 0 and num % print_interval == 0:
                    print(f'[{num}]', os.path.join(file_path))
    mean_rgb = sum_mean / (1. * num)
    std_rgb = sum_std / (1. * num)
    return mean_rgb, std_rgb, num


def mean_std_oct(root, print_interval=100):
    sum_mean = np.zeros(3)
    sum_std = np.zeros(3)

    num = 0
    for dir1 in os.listdir(root):
        for dir2 in os.listdir(os.path.join(root, dir1)):
            if dir2.endswith(extensions):
                continue
            for file in os.listdir(os.path.join(root, dir1, dir2)):
                if file.endswith(extensions):
                    file_path = os.path.join(root, dir1, dir2, file)
                    img = cv2.imread(file_path)
                    mean_b = np.mean(img[:, :, 0])
                    mean_g = np.mean(img[:, :, 1])
                    mean_r = np.mean(img[:, :, 2])
                    std_b = np.std(img[:, :, 0])
                    std_g = np.std(img[:, :, 1])
                    std_r = np.std(img[:, :, 2])

                    mean_rgb = np.array([mean_r, mean_g, mean_b])
                    std_rgb = np.array([std_r, std_g, std_b])
                    sum_mean = sum_mean + mean_rgb
                    sum_std = sum_std + std_rgb

                    num += 1
                    if print_interval > 0 and num % print_interval == 0:
                        print(f'[{num}]', os.path.join(file_path))
    mean_rgb = sum_mean / (1. * num)
    std_rgb = sum_std / (1. * num)
    return mean_rgb, std_rgb, num


def print_result(mean_rgb, std_rgb, num):
    print('------')
    print(f'Total images: {num}')
    print('RGB mean:', mean_rgb)
    print('RGB std:', std_rgb)
    print('RGB mean norm:', mean_rgb.astype(float) / 255.)
    print('RGB std norm:', std_rgb.astype(float) / 255.)
    print()
    print('For mmlab:')
    print('mean=[{:.3f}, {:.3f}, {:.3f}], # [R, G, B]'.format(mean_rgb[0], mean_rgb[1], mean_rgb[2]))
    print('std=[{:.3f}, {:.3f}, {:.3f}], # [R, G, B]'.format(std_rgb[0], std_rgb[1], std_rgb[2]))


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('-d', '--dir', type=str,
                      default='../data/Glaucoma_grading/training/multi-modality_images',
                    #   default='/home1/commonfile/CSURetina10K/SLOImages-v3.4',
                      help='path of imageset')
    args = args.parse_args()

    # results = mean_std(args.dir, print_interval=1)
    # print_result(*results)

    results1 = mean_std_fundus(args.dir, print_interval=1)
    results2 = mean_std_oct(args.dir, print_interval=1)

    print('fundus image')
    print_result(*results1)
    print('oct image')
    print_result(*results2)
