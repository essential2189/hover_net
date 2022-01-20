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


def mat2vein(mat_path, image_path):

    data_name = get_data_name(image_path)
    createFolder('../../output/' + data_name + '_vein')


    for mat, image in zip(os.listdir(mat_path), os.listdir(image_path)):
        images = loadmat(mat)

        inst_map = images['inst_map']

        inst_map = inst_map.astype('uint8')
        print(inst_map.shape)

        opencv_image = cv2.imread(image)

        inst_map_inv = 255 - inst_map
        inst_map_inv[inst_map_inv != 255] = 0

        inst_map_inv = inst_map_inv.astype(np.uint8)
        opencv_image = opencv_image.astype(np.uint8)
        print(opencv_image.shape, inst_map_inv.shape)
        print(opencv_image.dtype, inst_map_inv.dtype)

        inst_map_inv[inst_map_inv != 255] = 0
        image_bitwise = cv2.bitwise_and(opencv_image, opencv_image, mask=inst_map_inv)
        image_bitwise = cv2.cvtColor(image_bitwise, cv2.COLOR_BGR2RGB)

        cv2.imwrite('../../output/' + data_name + '_vein', image_bitwise)


def dbscan_filter(mat_path, output_path, image_path, kumar, eps, minsample):
    cnt_list = []
    # createFolder(output_path)

    mat_files = sorted(os.listdir(mat_path))
    img_files = sorted(os.listdir(image_path))


    for mat, image in zip(mat_files, img_files):
        #print(mat_files, img_files)
        data_name = mat.split('.')[0]
        mat_dict = loadmat(mat_path + mat)
        len_predict_red = 0
        len_predict_green = 0
        len_predict = 0
        len_predict_green_red_y = 0

        cell_num = len(mat_dict['inst_uid'])

        # kumar
        if kumar:  # matlab does not have None type array
            mat_file_value = mat_dict['inst_centroid']

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
                # save_path = "%s/overlay/%s.png" % (output_path, data_name)
                # cv_image = cv2.imread(str(image_path) + str(image))
                # cv2.imwrite(save_path, cv_image)


        # consep
        else:
            mat_file_value = mat_dict['inst_centroid']
            mat_file_type = mat_dict['inst_type']

            mat_file_value_green, mat_file_red, mat_file_blue, mat_file_y = anomaly_detection.cell_by_color(mat_file_value, mat_file_type)

            mat_file_green_red_y = np.vstack([mat_file_value_green])
            mat_file_green_red_y = mat_file_green_red_y[:, 1:3]


            if cell_num >= 90:
                if len(mat_file_green_red_y) > 0:
                    len_predict_green_red_y = anomaly_detection.dbscan_nonePD(mat_file_green_red_y, eps, minsample)

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



def main(eps):
    global min, min_s, min_e

    data = '20220112-17:55_CELL1101-1'
    img_data = 'CELL1101-1'
    csv_data = 'ann1101-1.csv'
    kumar = False

    if kumar:
        mat_path = '../../output/kumar/' + data + '/mat/'
        image_path = '../../datasets/image500/' + img_data + '/'
        output_path = '../../output/filter/kumar/' + data
        csv_path = '../../output/kumar/' + data + '/' + csv_data
    else:
        mat_path = '../../output/pannuke/' + data + '/mat/'
        image_path = '../../datasets/image500/' + img_data + '/'
        output_path = '../../output/filter/pannuke/' + data
        csv_path = '../../output/pannuke/' + data + '/' + csv_data


    for minsample in tqdm(list_minsample):
        cnt_list = dbscan_filter(mat_path, output_path, image_path, kumar, eps, minsample)

        try:
            name_list, id_list = check_ann(cnt_list, csv_path)

            if len(id_list) == 0:
                print('eps:', eps, ', min_sample:', minsample, ' -- ', len(name_list), len(id_list))
                if len(name_list) < min.value:
                    min.value = len(name_list)
                    min_s.value = minsample
                    min_e.value = eps

        except:
            if len(cnt_list) < min.value:
                print('(normal) eps:', eps, ', min_sample:', minsample, ' -- ', len(cnt_list))
                min.value = len(cnt_list)
                min_s.value = minsample
                min_e.value = eps

    # print(len(id_list))
    # print(name_list)
    # print(len(name_list))
    # print(min.value)
    # print(eps, min.value, min_e.value, min_s.value)


def init(arg1, arg2, arg3):
    global min, min_s, min_e
    min = arg1
    min_s = arg2
    min_e = arg3


list_eps = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
list_minsample = [2, 3, 4, 5, 6, 7, 8]


if __name__ == '__main__':
    start = time.time()
    print(multiprocessing.cpu_count())
    print(len(list_eps))
    min = Value('i', 9999)
    min_s = Value('i', 0)
    min_e = Value('i', 0)
    pool = multiprocessing.Pool(initializer=init, initargs=(min, min_s, min_e))
    pool.map(main, list_eps)
    pool.close()
    pool.join()

    print('eps:', min_e.value, ', min_sample:', min_s.value, ' -- ', min.value)


    #mat2vein(mat_path, image_path)
    end = time.time()
    print(datetime.timedelta(seconds=end-start))