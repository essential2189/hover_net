#-*-coding:utf-8-*-

import numpy as np
import cv2
import gc
from tqdm import tqdm

def watershed(opencv_image):
    top_n_label = 2

    gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
    print('convert gray end')

    gray[gray == 0] = 255

    _, cvt_img = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)
    del(gray)
    print('threshold end')

    # kernel = np.ones((3, 3), np.uint8)
    # cvt_img = cv2.morphologyEx(cvt_img, cv2.MORPH_OPEN, kernel, iterations=3)

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
    # for ins in tqdm(sort_label_list[:top_n_label]):
    # for ins in tqdm(sort_label_list):
    # result[markers == ins[0]] = 255
    ins = sort_label_list[0]
    result[markers == ins[0]] = 255

    result2 = np.zeros(markers.shape)
    ins2 = sort_label_list[1]
    result2[markers == ins2[0]] = 255

    print(result.shape)
    # print(result2.shape)
    print('top n label end')
    del(ret)
    del(markers)
    del(sort_label_list)
    del(label_dict)
    del(cvt_img)

    #
    # for y, ins in enumerate(result):
    #     for x, val in enumerate(ins):
    #         if val != 0:
    #             if min_x >= x:
    #                 min_x = x
    #             if max_x <= x:
    #                 max_x = x
    #             if min_y >= y:
    #                 min_y = y
    #             if max_y <= y:
    #                 max_y = y
    # print(min_x, max_x, min_y, max_y)
    # roi_result = result[min_y:max_y, min_x:max_x]

    return result, result2