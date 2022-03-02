from openslide import OpenSlide
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from skimage import io
import sys
from tqdm import tqdm
from PIL import Image
import get_xml

def check_dict(dict_, x, y, crop, min_x, max_x, min_y, max_y):
    anomaly = False

    for d in dict_.values():
        if x <= d[0]-min_x <= x + crop or x <= d[1]-min_x <= x + crop:
            if y <= d[2]-min_y <= y + crop or y <= d[3]-min_y <= y + crop:
                anomaly = True

    return anomaly


def crop_image(level_1, level_2, image, data_name, crop, min_x, max_x, min_y, max_y):
    filename_cnt = 0
    y = 0

    y_for = int(level_2 / crop)
    x_for = int(level_1 / crop)
    img_np = np.asarray(image)

    print('start crop')
    dict_ = get_xml.get_xml_dict('/home/sjwang/biotox/datasets/mrxs_label/1101-1/1101-1.xml')
    for i in tqdm(range(y_for)):
        x = 0
        for j in range(x_for):
            img_crop = img_np[y:y + crop, x:x + crop]

            anomaly = check_dict(dict_, x, y, crop, min_x, max_x, min_y, max_y)

            avg = (img_crop.mean(axis=0).mean(axis=0)[0] + img_crop.mean(axis=0).mean(axis=0)[1] + img_crop.mean(axis=0).mean(axis=0)[2]) / 3.0

            if avg < 254 and anomaly == True:
                print('anomaly')
                cv2.imwrite('../../datasets/image' + str(crop) + '/' + data_name + '/anomaly/img{}_({},{}).png'.format(filename_cnt, y, x), img_crop)
                filename_cnt += 1

            elif avg < 254 and anomaly == False:
                cv2.imwrite('../../datasets/image' + str(crop) + '/' + data_name + '/normal/img{}_({},{}).png'.format(filename_cnt, y, x), img_crop)
                filename_cnt += 1

            x += crop  # 50% 교차 crop
        y += crop  # 50% 교차 crop


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