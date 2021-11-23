from openslide import OpenSlide
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from skimage import io
import sys
from tqdm import tqdm
from PIL import Image

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
        print ('Error: Creating directory. ' +  directory)

def concat_tile(im_list_2d):
    return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])

def main(path):
    print(path)
    data_path = path
    image_path = path.split('/')
    data_name = image_path[3].split('.')[0]

    wsi = OpenSlide(data_path)
    print('open mrxs end')

    level_dim = wsi.dimensions
    level_1 = level_dim[0]
    level_2 = level_dim[1]
    level_1_ex = 0
    level_2_ex = 0

    img = wsi.read_region((0, 0), 0, (level_1, level_2))
    print('mrxs load end')

    while level_1 % 1000 != 0:
        level_1 += 1
        level_1_ex += 1
    while level_2 % 1000 != 0:
        level_2 += 1
        level_2_ex += 1
    print('level1 : ', level_1, ' level2 : ', level_2)
    print('level1_ex : ', level_1_ex, ' level2_ex : ', level_2_ex)
    print('cal extend end')

    image = add_margin(img, level_2_ex, level_1_ex, 0, 0, (255, 255, 255))
    ##
    image.save('../datasets/image/image.png', 'png', optimize=True)
    test_np = np.array(image)
    test_cv = cv2.cvtColor(test_np, cv2.COLOR_RGB2BGR)
    image_s = cv2.resize(test_cv, dsize=(0, 0), fx=0.1, fy=0.1)
    cv2.imwrite('../datasets/image/image_np.jpg', image_s)
    ##
    print('image padding end')
    print(image.size)
    filename_cnt = 0
    y = 0
    y_for = int(level_2 / 1000)
    x_for = int(level_1 / 1000)

    img_white = Image.new('RGB', (1000, 1000), (255, 255, 255))
    img_white_np = np.array(img_white)
    img_white_cv = cv2.cvtColor(img_white_np, cv2.COLOR_RGB2BGR)
    tile = [[] for i in range(y_for)]

    print('start crop')
    for i in tqdm(range(y_for)):
        x = 0
        for j in range(x_for):
            img_crop = image.crop((x, y, x+1000, y+1000))
            img_np = np.array(img_crop)
            img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            avg_rb = (img_np.mean(axis=0).mean(axis=0)[0] + img_np.mean(axis=0).mean(axis=0)[2]) / 2.0
            avg = (img_np.mean(axis=0).mean(axis=0)[0] + img_np.mean(axis=0).mean(axis=0)[1] + img_np.mean(axis=0).mean(axis=0)[2]) / 3.0

            if avg_rb > 180 and avg < 220:
                #createFolder('../datasets/image/'+data_name)
                #img_crop.save('../datasets/image/'+data_name+'/img_{}.png'.format(filename_cnt), 'png', optimize=True)
                im1_s = cv2.resize(img_cv, dsize=(0, 0), fx=0.1, fy=0.1)
                tile[i].append(im1_s)
                #filename_cnt += 1
            else:
                im1_s = cv2.resize(img_white_cv, dsize=(0, 0), fx=0.1, fy=0.1)
                tile[i].append(im1_s)

            x += 1000  # 50% 교차 crop
        y += 1000  # 50% 교차 crop


    im_tile = concat_tile(tile)
    cv2.imwrite('../datasets/image/opencv_concat_tile.jpg', im_tile)

if __name__ == '__main__':
    #print(sys.argv[1])
    main('../datasets/mrxs/CELL1101-1.mrxs')