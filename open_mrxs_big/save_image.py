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



def main(path):
    print(path)

    wsi = OpenSlide(path)
    print('open mrxs end')

    level_dim = wsi.dimensions
    level_1 = level_dim[0]
    level_2 = level_dim[1]

    img = wsi.read_region((0, 0), 0, (level_1, level_2))
    print('mrxs load end')

    img.save('../datasets/WSI/WSI/CELL1101-1.tif', format='tif')



if __name__ == '__main__':
    start = time.time()
    main('../../datasets/mrxs/CELL1101-1.mrxs')
    end = time.time()
    print(datetime.timedelta(seconds=end - start))