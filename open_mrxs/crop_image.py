from openslide import OpenSlide
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from skimage import io
import sys
from tqdm import tqdm
from PIL import Image


def crop_image(level_1, level_2, image, data_name, crop):
    filename_cnt = 0
    y = 0

    y_for = (int(level_2 / crop) * 2) - 1
    x_for = (int(level_1 / crop) * 2) - 1
    img_np = np.asarray(image)
    print('start crop')
    for i in tqdm(range(y_for)):
        x = 0
        for j in range(x_for):
            img_crop = img_np[y:y+crop, x:x+crop]
            # img_crop = cv2.cvtColor(img_crop, cv2.COLOR_RGBA2BGRA)

            avg_rb = (img_crop.mean(axis=0).mean(axis=0)[0] + img_crop.mean(axis=0).mean(axis=0)[2]) / 2.0
            avg = (img_crop.mean(axis=0).mean(axis=0)[0] + img_crop.mean(axis=0).mean(axis=0)[1] + img_crop.mean(axis=0).mean(axis=0)[2]) / 3.0

            if avg_rb > 180 and avg < 230:
                cv2.imwrite('../../datasets/image'+str(crop)+'/'+data_name+'/img_{}.png'.format(filename_cnt), img_crop)
                filename_cnt += 1

            x += crop//2  # 50% 교차 crop
        y += crop//2  # 50% 교차 crop


def check_crop(level_1, level_2, image, crop):
    y = 0
    y_for = int(level_2 / crop)
    x_for = int(level_1 / crop)

    img_white = Image.new('RGB', (crop, crop), (255, 255, 255))
    img_white_np = np.asarray(img_white)
    tile = [[] for _ in range(y_for)]

    img_np = np.asarray(image)

    print('start check crop')
    for i in tqdm(range(y_for)):
        x = 0
        for j in range(x_for):
            img_crop = img_np[y:y+crop, x:x+crop]

            avg_rb = (img_crop.mean(axis=0).mean(axis=0)[0] + img_crop.mean(axis=0).mean(axis=0)[2]) / 2.0
            avg = (img_crop.mean(axis=0).mean(axis=0)[0] + img_crop.mean(axis=0).mean(axis=0)[1] + img_crop.mean(axis=0).mean(axis=0)[2]) / 3.0

            if avg_rb > 180 and avg < 230:
                im1_s = cv2.resize(img_crop, dsize=(0, 0), fx=0.1, fy=0.1)
                tile[i].append(im1_s)

            else:
                im1_s = cv2.resize(img_white_np, dsize=(0, 0), fx=0.1, fy=0.1)
                tile[i].append(im1_s)

            x += crop
        y += crop

    return tile