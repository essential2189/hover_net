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

def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))

    return result


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


def background_white(img):
    x = np.array(img)
    r, g, b, a = np.rollaxis(x, axis=-1)
    r[a == 0] = 255
    g[a == 0] = 255
    b[a == 0] = 255
    x = np.dstack([r, g, b, a])
    img = Image.fromarray(x, 'RGBA')
    print('image background white end')

    return img


def cal_margin(level_1, level_2, crop):
    level_1_ex = 0
    level_2_ex = 0

    while level_1 % crop != 0:
        level_1 += 1
        level_1_ex += 1
    while level_2 % crop != 0:
        level_2 += 1
        level_2_ex += 1
    print('level1 : ', level_1, ' level2 : ', level_2)
    print('level1_ex : ', level_1_ex, ' level2_ex : ', level_2_ex)
    print('cal extend end')

    return level_1, level_2, level_1_ex, level_2_ex


def get_data_name(path):
    image_path = path.split('/')
    data_name = image_path[3].split('.')[0]

    return data_name


def main(path, crop):
    print(path)

    data_name = get_data_name(path)

    wsi = OpenSlide(path)
    print('open mrxs end')

    level_dim = wsi.dimensions
    level_1 = level_dim[0]
    level_2 = level_dim[1]

    img = wsi.read_region((0, 0), 0, (level_1, level_2))
    print('mrxs load end')

    level_1, level_2, level_1_ex, level_2_ex = cal_margin(level_1, level_2, crop)

    img_white = background_white(img)

    image = add_margin(img_white, level_2_ex, level_1_ex, 0, 0, (255, 255, 255, 255))
    print('add margin end')
    image.save('../datasets/'+data_name+'.png', 'PNG')
    print('image padding end')
    print(image.size)



if __name__ == '__main__':
    crop = 500
    main('../datasets/mrxs/CELL1101-1.mrxs', crop)