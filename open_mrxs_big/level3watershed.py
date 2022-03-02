from openslide import OpenSlide
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

from skimage import io
import sys
from tqdm import tqdm
from PIL import Image
import gc

import watershed
from open_mrxs import del_y_margin
from open_mrxs import add_margin


def min_image(img, level_1, level_2):
    pos = np.where(img == 255)

    min_x = min(pos[1])
    min_y = min(pos[0])
    max_x = max(pos[1])
    max_y = max(pos[0])

    return img[min_y:max_y, min_x:max_x], min_x, max_x, min_y, max_y


def dim_3(path):
    wsi = OpenSlide(path)
    print('3 open mrxs end')

    level_dim = wsi.level_dimensions
    level_1 = level_dim[5][0]
    level_2 = level_dim[5][1]

    # level_2_top, level_2_bot = del_y_margin(level_2, crop)

    img = wsi.read_region((0, 0), 5, (level_1, level_2))
    print(img.size[0], img.size[1])
    print('3 mrxs load end')

    # level_2_top, level_2_bot = del_y_margin(level_2, crop)
    # img = img.crop((0, level_2_top, level_1, level_2_bot))

    np_image = np.array(img)
    del(img)
    print('3 image numpy end')

    # level_2_top, level_2_bot = del_y_margin(level_2, crop)
    # np_image = np_image[level_2_top:level_2_bot, 0:level_1]

    # np_image, min_y, max_y = min_image(np_image, level_2)

    opencv_image = cv2.cvtColor(np_image, cv2.COLOR_RGBA2BGR)
    del(np_image)
    gc.collect()
    print('3 convert end')

    image, image2 = watershed.watershed(opencv_image)
    del(opencv_image)
    print('3 watershed end')

    image, min_x, max_x, min_y, max_y = min_image(image, level_1, level_2)

    image2, min_x2, max_x2, min_y2, max_y2 = min_image(image2, level_1, level_2)

    min_x = int(min_x * (level_dim[0][0] / level_dim[5][0]))
    max_x = int(max_x * (level_dim[0][0] / level_dim[5][0]))

    min_y = int(min_y * (level_dim[0][1] / level_dim[5][1]))
    max_y = int(max_y * (level_dim[0][1] / level_dim[5][1]))

    min_x2 = int(min_x2 * (level_dim[0][0] / level_dim[5][0]))
    max_x2 = int(max_x2 * (level_dim[0][0] / level_dim[5][0]))

    min_y2 = int(min_y2 * (level_dim[0][1] / level_dim[5][1]))
    max_y2 = int(max_y2 * (level_dim[0][1] / level_dim[5][1]))

    img = wsi.read_region((min_x, min_y), 2, (max_x - min_x, max_y - min_y))
    img2 = wsi.read_region((min_x2, min_y2), 2, (max_x2 - min_x2, max_y2 - min_y2))

    img.save('../../datasets' + '/2.png', 'png')
    img2.save('../../datasets' + '/3.png', 'png')

    return image, image2, min_x, max_x, min_y, max_y, min_x2, max_x2, min_y2, max_y2


# def make_mask(bitnot):
#     gray = cv2.cvtColor(bitnot, cv2.COLOR_BGR2GRAY)
#     th, threshed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     del(gray)
#     del(th)
#
#     return threshed


def watershed2mask(path, level_1):
    image, image2, min_x, max_x, min_y, max_y, min_x2, max_x2, min_y2, max_y2 = dim_3(path)
    print('3 water dim3 end')

    # water_dim3 = cv2.bitwise_not(water_dim3)
    # print('3 water dim3 bitwise not end')

    # mask = make_mask(water_dim3)
    # del(water_dim3)
    # print('3 make mask end')

    mask = cv2.resize(image, dsize=(max_x-min_x, max_y-min_y), interpolation=cv2.INTER_CUBIC)
    mask2 = cv2.resize(image2, dsize=(max_x2 - min_x2, max_y2 - min_y2), interpolation=cv2.INTER_CUBIC)
    del(image)
    del(image2)
    print(mask.shape)
    print('3 water dim3 resize end')
    # cv2.imwrite('../../datasets/image500' + '/3_watershed.png', mask)

    return mask, mask2, min_x, max_x, min_y, max_y, min_x2, max_x2, min_y2, max_y2