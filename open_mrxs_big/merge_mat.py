import matplotlib.pyplot as plt
import numpy as np
import h5py
from PIL import Image
import os
import scipy
from scipy.io import loadmat
from openslide import OpenSlide
import os
import cv2
from skimage import io
import sys
from tqdm import tqdm
from PIL import Image
import gc
import time
import datetime

import open_mrxs


mat = loadmat('pred/mat/img_1355.mat')
mat2 = loadmat('pred/mat/img_1894.mat')

inst_map = mat['inst_map']
centroid = mat['inst_centroid']

inst_map2 = mat2['inst_map']
centroid2 = mat2['inst_centroid']

print(inst_map.shape, inst_map2.shape)
inst_map_ = np.hstack([inst_map, inst_map2])

plt.figure(figsize=(12, 12))
plt.imshow(inst_map_)

def main(path, crop):
    print(path)

    wsi = OpenSlide(path)
    print('open mrxs end')

    level_dim = wsi.level_dimensions
    x = level_dim[0][0]
    y = level_dim[0][1]

    x_x, x_ex, y_y, y_ex = open_mrxs.cal_margin(x, y, crop)

    x_for = int(x_x / crop)
    y_for = int(y_y / crop)

    mat_path = '../../output'
    mat_list = sorted(os.listdir(mat_path))
    mat_ = loadmat(mat_list[0])

    map_merge_x = mat_['inst_map']
    centroid_merge = mat_['inst_centroid']
    type_merge = mat_['inst_type']
    uid_merge = mat_['inst_uid']

    for i in range(y_for):
        for j, mat in zip(range(x_for), mat_list):
            mat = loadmat(mat)

            inst_map = mat['inst_map']
            inst_centroid = mat['inst_centroid']
            inst_type = mat['inst_type']
            inst_uid = mat['inst_uid']

            map_merge_x = np.hstack([map_merge_x, inst_map])
            centroid_merge = np.vstarck([centroid_merge, inst_centroid])
            type_merge = np.vstarck([type_merge, inst_type])
            uid_merge = np.vstarck([uid_merge, inst_uid])

        if i == 0:
            map_merge_y = map_merge_x
        else:
            map_merge_y = np.vstack([map_merge_y, map_merge_x])

    temp = {'inst_map': map_merge_y, 'inst_uid': uid_merge, 'inst_type': type_merge, 'inst_centroid': centroid_merge}

    scipy.io.savemat('test.mat', temp)



if __name__ == '__main__':
    start = time.time()
    crop = 8000
    main('../../datasets/mrxs_a/CELL1101-1.mrxs', crop)
    end = time.time()
    print(datetime.timedelta(seconds=end-start))