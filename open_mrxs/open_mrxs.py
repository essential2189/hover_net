from openslide import OpenSlide
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = str(2**64)
import cv2
from skimage import io
import sys
from tqdm import tqdm
from PIL import Image
import gc
import time
import datetime

import crop_image
import level3watershed
import openslide


Image.MAX_IMAGE_PIXELS = None


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


def concat_tile(im_list_2d):
    return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])


def background_white(x):
    #x = np.array(img)
    r, g, b, a = np.rollaxis(x, axis=-1)
    r[a == 0] = 255
    g[a == 0] = 255
    b[a == 0] = 255
    x = np.dstack([r, g, b, a])
    #img = Image.fromarray(x, 'RGBA')
    print('image background white end')

    return x


def cal_margin(x, y, crop):
    x_ex = 0
    y_ex = 0

    while x % crop != 0:
        x += 1
        x_ex += 1

    while y % crop != 0:
        y += 1
        y_ex += 1

    print('level1 : ', x, ' level1_ex : ', x_ex)
    print('level2 : ', y, ' level2_ex : ', y_ex)
    print('cal extend end')

    return x, x_ex, y, y_ex


def del_y_margin(y, crop):
    y_top = int(y * 0.2)
    y_bot = int(y * 0.9)

    # while y_top % crop != 0:
    #     y_top += 1
    #
    # while y_bot % crop != 0:
    #     y_bot += 1

    print('level2_top : ', y_top, 'level2_bot : ', y_bot)
    print('cal del y end')

    return y_top, y_bot


def get_data_name(path):
    image_path = path.split('/')
    data_name = image_path[6].split('.')[0]

    return data_name


# def pixel_white(x, y, image):
#     for y in tqdm(range(y)):
#         for x in range(x):
#             r, g, b = image.getpixel((x, y))
#             avg = r + g + b
#             if avg == 0:
#                 image.putpixel((x, y), (255, 255, 255))
#
#     return image


def main(path, crop):
    print(path)

    data_name = get_data_name(path)

    createFolder('../../datasets/image' + str(crop) + '/' + data_name +'/anomaly')
    createFolder('../../datasets/image' + str(crop) + '/' + data_name +'/normal')

    wsi = OpenSlide(path)
    print('open mrxs end')
    print(wsi.level_downsamples)

    level_dim = wsi.level_dimensions
    x = level_dim[0][0]
    y = level_dim[0][1]

    # img = wsi.read_region((0, 0), 0, (x, y))
    # img.save('../../datasets/image' + str(crop) + '/'+ data_name+'.png', 'png', optimize=True)

    mask, min_x, max_x, min_y, max_y = level3watershed.watershed2mask(path, x)


    img = wsi.read_region((min_x, min_y), 0, (max_x-min_x, max_y-min_y))
    print(img.size[0], img.size[1])
    print('mrxs load end')

    img = np.array(img)

    img_resize = cv2.resize(img, dsize=(0, 0), fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)
    cv2.imwrite('../../datasets/'+ data_name + '.png', img_resize)
    del(img_resize)

    opencv_image = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    del(img)
    print('image to cv2 end')


    mask = mask.astype(np.uint8)
    opencv_image = opencv_image.astype(np.uint8)
    print(opencv_image.shape, mask.shape)
    print(opencv_image.dtype, mask.dtype)

    image_bitwise = cv2.bitwise_and(opencv_image, opencv_image, mask=mask)
    del(opencv_image)
    print(image_bitwise.size)
    print('biwise end')

    image_bitwise[mask == 0] = 255

    img_resize = cv2.resize(image_bitwise, dsize=(0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)
    cv2.imwrite('../../datasets/' + data_name + '.png', img_resize)
    del (img_resize)
    #
    # del(mask)
    # print('bitwise image white end')
    #
    # img = Image.fromarray(image_bitwise)
    # print('cv2 to image end')
    #
    # x = img.size[0]
    # y = img.size[1]
    #
    # x_x, x_ex, y_y, y_ex = cal_margin(x, y, crop)
    # img = add_margin(img, 0, x_ex, y_ex, 0, (255, 255, 255))
    # del(image_bitwise)
    # print('image padding end')
    # print(img.size)
    # #img.save('../../datasets/image' + str(crop) + '/add_margin.png', 'png', optimize=True)
    #
    # x = img.size[0]
    # y = img.size[1]
    #
    # crop_image.crop_image(x, y, img, data_name, crop, min_x, max_x, min_y, max_y)
    #
    # tile = crop_image.check_crop(x, y, img, crop)
    # del(img)
    # im_tile = concat_tile(tile)
    # del(tile)
    # cv2.imwrite('../../datasets/image'+str(crop)+'/'+data_name+'_opencv_concat_tile.jpg', im_tile)


from glob import glob
if __name__ == '__main__':
    start = time.time()
    #print(sys.argv[1])
    crop = 500
    for data in glob('/home/sjwang/biotox/datasets/mrxs3/*.mrxs'):
        main(data, crop)
    end = time.time()
    print(datetime.timedelta(seconds=end-start))

