import numpy as np
import cv2
import gc
from tqdm import tqdm
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
import time
import datetime
import glob

Image.MAX_IMAGE_PIXELS = None

def min_image(img, level_1, level_2):
    min_y = level_2
    max_y = 0
    min_x = level_1
    max_x = 0

    print('cut min image start')
    for y, ins in tqdm(enumerate(img)):
        for x, val in enumerate(ins):
            if val.any() != 0:
                if min_x >= x:
                    min_x = x
                if max_x <= x:
                    max_x = x

                if min_y >= y:
                    min_y = y
                if max_y <= y:
                    max_y = y

    print(min_x, max_x, min_y, max_y)
    print(img[min_y:max_y, min_x:max_x].shape)

    return img[min_y:max_y, min_x:max_x], min_x, max_x, min_y, max_y

def watershed(opencv_image):
    top_n_label = 2

    gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
    print('convert gray end')

    gray[gray == 0] = 255

    _, cvt_img = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)
    del(gray)
    print('threshold end')


    ret, markers = cv2.connectedComponents(cvt_img)
    print('connectedComponents end')

    label_dict = dict()
    for i in tqdm(range(ret)):
        if i == 0:
            continue
        label_dict[i] = len(markers[markers == i])
    sort_label_list = sorted(label_dict.items(), key=lambda item: item[1], reverse=True)
    print('label end')

    result = np.zeros(markers.shape)
    for ins in sort_label_list[:top_n_label]:
        result[markers == ins[0]] = 255
    print(result.shape)
    print('top n label end')
    del(ret)
    del(markers)
    del(sort_label_list)
    del(label_dict)
    del(cvt_img)

    return result


def dim_3(path):
    wsi = OpenSlide(path)
    print('3 open mrxs end')

    level_dim = wsi.level_dimensions
    print(level_dim)
    level_1 = level_dim[0][0]
    level_2 = level_dim[0][1]


    img = wsi.read_region((0, 0), 0, (level_1, level_2))
    print(img.size[0], img.size[1])
    print('3 mrxs load end')


    np_image = np.array(img)
    print(np_image.shape)
    del(img)
    print('3 image numpy end')

    np_image = cv2.resize(np_image, dsize=(0, 0), fx=0.01, fy=0.01, interpolation=cv2.INTER_AREA)

    opencv_image = cv2.cvtColor(np_image, cv2.COLOR_RGBA2BGR)
    del(np_image)
    gc.collect()
    print('3 convert end')

    image = watershed(opencv_image)
    del(opencv_image)
    print('3 watershed end')

    image, min_x, max_x, min_y, max_y = min_image(image, level_1, level_2)

    # min_x = int(min_x * (level_dim[0][0] / level_dim[5][0]))
    # max_x = int(max_x * (level_dim[0][0] / level_dim[5][0]))
    #
    # min_y = int(min_y * (level_dim[0][1] / level_dim[5][1]))
    # max_y = int(max_y * (level_dim[0][1] / level_dim[5][1]))

    return image, level_1, level_2


def main(path, data_name):
    print(path)

    # wsi = OpenSlide(path)
    # print('open mrxs end')
    #
    # level_dim = wsi.level_dimensions
    # x = level_dim[0][0]
    # y = level_dim[0][1]

    water_dim3, x, y = dim_3(path)

    water_dim3 = cv2.resize(water_dim3, dsize=(x, y), interpolation=cv2.INTER_CUBIC)

    # print(mask.shape)

    cv2.imwrite('../../datasets/WSI/mask/' + data_name + '.png', water_dim3)
    del(water_dim3)


if __name__ == '__main__':
    for data_name in tqdm(os.listdir('../../datasets/WSI/WSI/')):
        start = time.time()
        data_name = data_name.split('.')[0]
        main('../../datasets/WSI/WSI/' + data_name + '.tif', data_name)
        gc.collect()
        end = time.time()
        print(datetime.timedelta(seconds=end-start))

