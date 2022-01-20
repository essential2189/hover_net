import multiprocessing
from multiprocessing import Value

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import scipy
from scipy.io import loadmat
import time
import datetime

import cv2
import anomaly_detection
from tqdm import tqdm
import seaborn as sns
import matplotlib.cm as cm
from scipy.ndimage.filters import gaussian_filter
import io
from check_ann import check_ann

plt.rcParams.update({'figure.max_open_warning': 0})


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


def get_data_name(path):
    image_path = path.split('/')
    data_name = image_path[4]

    return data_name


def make_heatmap(x, y, s, bins=500):
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
    heatmap = gaussian_filter(heatmap, sigma=s)

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return heatmap.T, extent


def mat2cell(mat_path, image_path, img_data):
    createFolder('../../output/' + img_data + '_cell')

    mat_files = sorted(os.listdir(mat_path))
    img_files = sorted(os.listdir(image_path))

    for mat, image in tqdm(zip(mat_files, img_files)):
        mat_dict = loadmat(mat_path + mat)

        inst_map = mat_dict['inst_map']

        opencv_image = cv2.imread(image_path + image)

        inst_map = inst_map.astype(np.uint8)
        opencv_image = opencv_image.astype(np.uint8)
        # print(opencv_image.shape, inst_map.shape)
        # print(opencv_image.dtype, inst_map.dtype)

        # gray = cv2.cvtColor(inst_map, cv2.COLOR_BGR2GRAY)

        image_bitwise = cv2.bitwise_and(opencv_image, opencv_image, mask=inst_map)

        cv2.imwrite('../../output/' + img_data + '_cell/' + image, image_bitwise)


def mat2vein(mat_path, image_path, img_data):

    createFolder('../../output/' + img_data + '_vacuolation')

    mat_files = sorted(os.listdir(mat_path))
    img_files = sorted(os.listdir(image_path))


    for mat, image in tqdm(zip(mat_files, img_files)):
        mat_dict = loadmat(mat_path + mat)

        inst_map = mat_dict['inst_map']

        inst_map = inst_map.astype('uint8')
        # print(inst_map.shape)

        opencv_image = cv2.imread(image_path + image)

        inst_map_inv = 255 - inst_map
        inst_map_inv[inst_map_inv != 255] = 0

        inst_map_inv = inst_map_inv.astype(np.uint8)
        opencv_image = opencv_image.astype(np.uint8)
        # print(opencv_image.shape, inst_map_inv.shape)
        # print(opencv_image.dtype, inst_map_inv.dtype)

        inst_map_inv[inst_map_inv != 255] = 0
        image_bitwise = cv2.bitwise_and(opencv_image, opencv_image, mask=inst_map_inv)

        cv2.imwrite('../../output/' + img_data + '_vacuolation/' + image, image_bitwise)


def dbscan_filter(mat_path, output_path, image_path, kumar, eps, minsample):
    cnt_list = []
    # createFolder(output_path)

    mat_files = sorted(os.listdir(mat_path))
    img_files = sorted(os.listdir(image_path))


    for mat, image in tqdm(zip(mat_files, img_files)):
        #print(mat_files, img_files)
        data_name = mat.split('.')[0]
        mat_dict = loadmat(mat_path + mat)
        len_predict_red = 0
        len_predict_green = 0
        len_predict = 0

        cell_num = len(mat_dict['inst_uid'])

        # kumar
        if kumar:  # matlab does not have None type array
            mat_dict.pop("inst_type", None)

            mat_file_value = mat_dict['inst_centroid']

            # x = mat_file_value[:, 0]
            # y = mat_file_value[:, 1]

            if cell_num >= 50:
                if len(mat_file_value) > 0:
                    len_predict = anomaly_detection.dbscan_nonePD(mat_file_value, eps, minsample)

                    if len_predict > 1:
                        cnt_list.append(mat)

                # save_path = "%s/dbscan/%s.png" % (output_path, data_name)
                # sns.pairplot(g, hue='predict', height=6, kind='scatter', diag_kind='hist')
                # plt.savefig(save_path)
                # plt.clf()
                # plt.cla()
                #
                # save_path = "%s/csv/%s.csv" % (output_path, data_name)
                # g.to_csv(save_path, mode='w')
                #
                    save_path = "%s/overlay/%s.png" % (output_path, data_name)
                    cv_image = cv2.imread(str(image_path) + str(image))
                    cv2.imwrite(save_path, cv_image)


        # consep
        else:
            mat_file_value = mat_dict['inst_centroid']
            mat_file_type = mat_dict['inst_type']

            mat_file_value_green, mat_file_red, mat_file_blue, mat_file_y = anomaly_detection.cell_by_color(mat_file_value, mat_file_type)

            # mat_file_green_red_y = np.vstack([mat_file_red, mat_file_value_green, mat_file_y])
            # mat_file_green_red_y = mat_file_green_red_y[:, 1:3]
            mat_file_value_green = mat_file_value_green[:, 1:3]

            if cell_num >= 90:
                if len(mat_file_value_green) > 0:
                    len_predict_green_red_y = anomaly_detection.dbscan_nonePD(mat_file_value_green, eps, minsample)

                    if len_predict_green_red_y > 1:
                        cnt_list.append(mat)
                    # print(len(cnt_list))
                    # save_path = "%s/gr_dbscan/%s.png" % (output_path, data_name)
                    # sns.pairplot(gr, hue='predict', height=6, kind='scatter', diag_kind='hist')
                    # plt.savefig(save_path)
                    # plt.clf()
                    # plt.cla()
                    #
                    # save_path = "%s/gr_csv/%s.csv" % (output_path, data_name)
                    # gr.to_csv(save_path, mode='w')
                    #
                    # save_path = "%s/gr_overlay/%s.png" % (output_path, data_name)
                    # cv_image = cv2.imread(str(image_path) + str(image))
                    # cv2.imwrite(save_path, cv_image)

    return cnt_list



def folder(kumar, output_path):
    if kumar :
        createFolder(output_path + '/dbscan')
        createFolder(output_path + '/csv')
        createFolder(output_path + '/overlay')
    else :
        # createFolder(output_path + '/g_dbscan')
        # createFolder(output_path + '/g_csv')
        # createFolder(output_path + '/g_overlay')
        # createFolder(output_path + '/r_dbscan')
        # createFolder(output_path + '/r_csv')
        # createFolder(output_path + '/r_overlay')
        createFolder(output_path + '/gr_dbscan')
        createFolder(output_path + '/gr_csv')
        createFolder(output_path + '/gr_overlay')


if __name__ == '__main__':
    start = time.time()

    data = '20220111-17:59_CELL1507-1'
    img_data = 'CELL1507-1'
    csv_data = 'CELL1507-1.csv'
    kumar = True

    if kumar:
        mat_path = '../../output/kumar/' + data + '/mat/'
        image_path = '../../datasets/image500/' + img_data + '/'
        output_path = '../../output/filter/kumar/' + data
        csv_path = '../../output/kumar/' + data + '/' + csv_data
    else:
        mat_path = '../../output/consep/' + data + '/mat/'
        image_path = '../../datasets/image500/' + img_data + '/'
        output_path = '../../output/filter/consep/' + data
        csv_path = '../../output/consep/' + data + '/' + csv_data

    folder(kumar, output_path)

    if kumar:
        eps = 66
        minsample = 13
    else:
        eps = 60
        minsample = 11

    # cnt_list = dbscan_filter(mat_path, output_path, image_path, kumar, eps, minsample)
    #
    # name_list, id_list = check_ann(cnt_list, csv_path)
    #
    #
    # print(len(name_list), len(id_list))
    # print(name_list)


    mat2vein(mat_path, image_path, img_data)

    mat2cell(mat_path, image_path, img_data)


    end = time.time()
    print(datetime.timedelta(seconds=end-start))