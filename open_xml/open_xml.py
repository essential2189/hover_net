import xml.etree.ElementTree as elemTree
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



def get_data_name(path):
    image_path = path.split('/')
    data_name = image_path[4].split('.')[0]

    return data_name

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


def get_xml(tree, img, data_name):
    root = tree.getroot()
    destination = root.findall("destination/annotations/annotation")

    # for x in destination:
    #     print(x.attrib['name'])
    #
    # print(x_y[0])
    # print(destination[0].attrib['name'])

    cnt = 0
    for x in tqdm(destination):
        point = []
        name_ = x.attrib['name']
        for a in x.findall('p'):
            temp = []
            temp.append(int(a.attrib['x']))
            temp.append(int(a.attrib['y']))
            point.append(temp)

        point = np.array(point)

        x_min = np.sort(point[:, 0], axis=0)[0]
        x_max = np.sort(point[:, 0], axis=0)[-1]
        y_min = np.sort(point[:, 1], axis=0)[0]
        y_max = np.sort(point[:, 1], axis=0)[-1]

        img_save = img[y_min:y_max, x_min:x_max]

        cv2.imwrite('/home/sjwang/biotox/datasets/mrxs_label/' + data_name + '/' + name_ + '_' + str(cnt) + '.png', img_save)
        cnt += 1

def load_wsi(path):
    wsi = OpenSlide(path)

    level_dim = wsi.level_dimensions
    x = level_dim[0][0]
    y = level_dim[0][1]

    img = wsi.read_region((0, 0), 0, (x, y))
    print('mrxs load end')

    np_img = np.array(img)
    del(img)
    # np_img[np_img == 0] = 255
    print('numpy end')

    np_img = cv2.cvtColor(np_img, cv2.COLOR_RGBA2BGR)

    return np_img

def main(path):
    print(path)

    data_name = get_data_name(path)
    createFolder('../../datasets/mrxs_label/' + data_name)

    img = load_wsi(path)

    tree = elemTree.parse('/home/sjwang/biotox/datasets/mrxs_label/1101-1/1101-1.xml')
    get_xml(tree, img, data_name)


if __name__ == '__main__':
    start = time.time()
    main('../../datasets/mrxs_a/CELL1101-1.mrxs')
    end = time.time()
    print(datetime.timedelta(seconds=end-start))