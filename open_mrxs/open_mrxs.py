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

import crop_image
import level3watershed


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


def cal_margin(level_1, level_2, crop):
    level_1_ex = 0
    level_2_ex = 0

    while level_1 % crop != 0:
        level_1 += 1
        level_1_ex += 1

    while level_2 % crop != 0:
        level_2 += 1
        level_2_ex += 1

    print('level1 : ', level_1, ' level1_ex : ', level_1_ex)
    print('level2 : ', level_2, ' level2_ex : ', level_2_ex)
    print('cal extend end')

    return level_1, level_1_ex, level_2, level_2_ex


def del_y_margin(level_2, crop):
    level_2_top = int(level_2 * 0.2)
    level_2_bot = int(level_2 * 0.9)

    # while level_2_top % crop != 0:
    #     level_2_top += 1
    #
    # while level_2_bot % crop != 0:
    #     level_2_bot += 1

    print('level2_top : ', level_2_top, 'level2_bot : ', level_2_bot)
    print('cal del y end')

    return level_2_top, level_2_bot


def get_data_name(path):
    image_path = path.split('/')
    data_name = image_path[4].split('.')[0]

    return data_name


# def pixel_white(level_1, level_2, image):
#     for y in tqdm(range(level_2)):
#         for x in range(level_1):
#             r, g, b = image.getpixel((x, y))
#             avg = r + g + b
#             if avg == 0:
#                 image.putpixel((x, y), (255, 255, 255))
#
#     return image


def main(path, crop):
    print(path)

    data_name = get_data_name(path)

    createFolder('../../datasets/image' + str(crop) + '/' + data_name)

    wsi = OpenSlide(path)
    print('open mrxs end')

    level_dim = wsi.level_dimensions
    level_1 = level_dim[0][0]
    level_2 = level_dim[0][1]

    mask, min_x, max_x, min_y, max_y = level3watershed.watershed2mask(path, level_1)
    print(min_x, max_x, min_y, max_y)

    img = wsi.read_region((min_x, min_y), 0, (max_x-min_x, max_y-min_y))
    #img.save('../../datasets/image' + str(crop) + '/wsi.png', 'png', optimize=True)
    print(img.size[0], img.size[1])
    print('mrxs load end')

    # level_2_top, level_2_bot = del_y_margin(level_2, crop)
    # img = img.crop((0, level_2_top, level_1, level_2_bot))

    img = np.array(img)

    # img = min_image(img, level_2)

    #img = background_white(img)

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

    # level_1 = image_bitwise.shape[1]
    # level_2 = image_bitwise.shape[0]

    # level_2_top, level_2_bot = del_y_margin(level_2, crop)
    # image_bitwise = image_bitwise[level_2_top:level_2_bot, 0:level_1]
    # image_bitwise = min_image(image_bitwise, level_2)
    # print(image_bitwise.size)
    # print('image y cut end')

    image_bitwise[mask == 0] = 255
    #cv2.imwrite('../../datasets/image' + str(crop) + '/mask.png', image_bitwise)
    del(mask)
    print('bitwise image white end')

    # image_bitwise = cv2.cvtColor(image_bitwise, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(image_bitwise)
    print('cv2 to image end')

    level_1 = img.size[0]
    level_2 = img.size[1]

    level_1_x, level_1_ex, level_2_y, level_2_ex = cal_margin(level_1, level_2, crop)
    img = add_margin(img, 0, level_1_ex, level_2_ex, 0, (255, 255, 255))
    del(image_bitwise)
    print('image padding end')
    print(img.size)
    #img.save('../../datasets/image' + str(crop) + '/add_margin.png', 'png', optimize=True)

    level_1 = img.size[0]
    level_2 = img.size[1]

    crop_image.crop_image(level_1, level_2, img, data_name, crop)

    tile = crop_image.check_crop(level_1, level_2, img, crop)
    del(img)
    im_tile = concat_tile(tile)
    del(tile)
    cv2.imwrite('../../datasets/image'+str(crop)+'/'+data_name+'_opencv_concat_tile.jpg', im_tile)


if __name__ == '__main__':
    start = time.time()
    #print(sys.argv[1])
    crop = 500
    main('../../datasets/mrxs3/2509-1.mrxs', crop)
    end = time.time()
    print(datetime.timedelta(seconds=end-start))

