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

    img = wsi.read_region((0, 0), 5, (level_1, level_2))

    print(img.size[0], img.size[1])
    print('3 mrxs load end')


    np_image = np.array(img)
    del(img)
    print('3 image numpy end')

    opencv_image = cv2.cvtColor(np_image, cv2.COLOR_RGBA2BGR)
    del(np_image)
    gc.collect()
    print('3 convert end')

    image = watershed.watershed(opencv_image)
    del(opencv_image)
    print('3 watershed end')

    image, min_x, max_x, min_y, max_y = min_image(image, level_1, level_2)


    min_x = int(min_x * (level_dim[0][0] / level_dim[5][0]))
    max_x = int(max_x * (level_dim[0][0] / level_dim[5][0]))

    min_y = int(min_y * (level_dim[0][1] / level_dim[5][1]))
    max_y = int(max_y * (level_dim[0][1] / level_dim[5][1]))


    return image, min_x, max_x, min_y, max_y


def watershed2mask(path, level_1):
    image1, min_x, max_x, min_y, max_y = dim_3(path)
    print('3 water dim3 end')

    mask = cv2.resize(image1, dsize=(max_x-min_x, max_y-min_y), interpolation=cv2.INTER_CUBIC)
    del(image1)

    print(mask.shape)
    print('3 water dim3 resize end')
    # cv2.imwrite('../../datasets/image500' + '/3_watershed.png', mask)

    return mask, min_x, max_x, min_y, max_y