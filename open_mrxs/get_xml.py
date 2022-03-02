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



def get_xml_dict(path):
    tree = elemTree.parse(path)

    root = tree.getroot()
    destination = root.findall("destination/annotations/annotation")

    x_min_list = []
    x_max_list = []
    y_min_list = []
    y_max_list = []

    dict_ = {}

    for x in destination:
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

        # x_min_list.append(x_min)
        # x_max_list.append(x_max)
        # y_min_list.append(y_min)
        # y_max_list.append(y_max)

        dict_[name_] = x_min, x_max, y_min, y_max

    return dict_
