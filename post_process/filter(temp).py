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

plt.rcParams.update({'figure.max_open_warning': 0})

cnt = 0
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


def dbscan_filter(mat_path, output_path, image_path, kumar):
    createFolder(output_path)

    mat_files = sorted(os.listdir(mat_path))
    img_files = sorted(os.listdir(image_path))

    for mat, image in tqdm(zip(mat_files, img_files)):
        print(mat, image)
        data_name = mat.split('.')[0]
        mat_dict = loadmat(mat_path + mat)
        len_predict_red = 0
        len_predict_green = 0
        len_predict = 0

        # kumar
        if kumar:  # matlab does not have None type array
            mat_dict.pop("inst_type", None)

            mat_file_value = mat_dict['inst_centroid']

            x = mat_file_value[:, 0]
            y = mat_file_value[:, 1]

            g, len_predict = anomaly_detection.dbscan_kumar(mat_file_value)

            if len_predict > 1:
                save_path = "%s/dbscan/%s.png" % (output_path, data_name)
                sns.pairplot(g, hue='predict', height=6, kind='scatter', diag_kind='hist')
                plt.savefig(save_path)
                plt.clf()
                plt.cla()

                save_path = "%s/csv/%s.csv" % (output_path, data_name)
                g.to_csv(save_path, mode='w')

                save_path = "%s/overlay/%s.png" % (output_path, data_name)
                cv_image = cv2.imread(str(image_path) + str(image))
                cv2.imwrite(save_path, cv_image)

                # save_path = "%s/overlay/%s.png" % (output_path, data_name)
                # cv_image = cv2.imread(str(image_path) + str(image))
                # s = 16
                # img, extent = make_heatmap(x, y, s)
                # plt.imshow(cv_image)
                # plt.imshow(img, extent=extent, origin='lower', cmap=cm.jet, alpha=0.3)
                # plt.savefig(save_path, format="png", bbox_inches='tight', pad_inches=0)
                # plt.clf()
                # plt.cla()

                # save_path = "%s/overlay/%s.png" % (output_path, data_name)
                # print(image)
                # cv_image = cv2.imread(str(image_path) + str(image))
                # s = 16
                # img, extent = make_heatmap(x, y, s)
                # norm = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                # norm = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
                # # norm = cv2.flip(norm, 0)
                # # norm = cv2.flip(norm, 1)
                # dst = cv2.addWeighted(cv_image, 0.7, norm, 0.3, 0)
                # cv2.imwrite(save_path, dst)

        # consep
        else:
            global cnt
            mat_file_value = mat_dict['inst_centroid']
            mat_file_type = mat_dict['inst_type']

            mat_file_value_green, mat_file_red, mat_file_blue, mat_file_y = anomaly_detection.cell_by_color(mat_file_value, mat_file_type)

            mat_file_green_red_y = np.vstack([mat_file_red, mat_file_value_green, mat_file_y])

            # if len(mat_file_value_green) > 0:
            #     g, len_predict_green = anomaly_detection.dbscan_green(mat_file_value_green)
            #
            # if len(mat_file_red) > 0:
            #     r, len_predict_red = anomaly_detection.dbscan_red(mat_file_red)

            if len(mat_file_green_red_y) > 0:
                gr, len_predict_green_red_y = anomaly_detection.dbscan_green_red(mat_file_green_red_y)

            if len_predict_green_red_y > 1:
                cnt += 1
                save_path = "%s/gr_dbscan/%s.png" % (output_path, data_name)
                sns.pairplot(gr, hue='predict', height=6, kind='scatter', diag_kind='hist')
                plt.savefig(save_path)
                plt.clf()
                plt.cla()

                save_path = "%s/gr_csv/%s.csv" % (output_path, data_name)
                gr.to_csv(save_path, mode='w')

                save_path = "%s/gr_overlay/%s.png" % (output_path, data_name)
                cv_image = cv2.imread(str(image_path) + str(image))
                cv2.imwrite(save_path, cv_image)

            # if len_predict_green > 1:
            #     x = mat_file_value_green[:, 0]
            #     y = mat_file_value_green[:, 1]
            #
            #     save_path = "%s/g_dbscan/%s.png" % (output_path, data_name)
            #     sns.pairplot(g, hue='predict', height=6, kind='scatter', diag_kind='hist')
            #     plt.savefig(save_path)
            #     plt.clf()
            #     plt.cla()
            #
            #     save_path = "%s/g_csv/%s.csv" % (output_path, data_name)
            #     g.to_csv(save_path, mode='w')
            #
            #     save_path = "%s/g_overlay/%s.png" % (output_path, data_name)
            #     cv_image = cv2.imread(str(image_path) + str(image))
            #     cv2.imwrite(save_path, cv_image)
            #
            # if len_predict_red > 1:
            #     x = mat_file_red[:, 0]
            #     y = mat_file_red[:, 1]
            #
            #     save_path = "%s/r_dbscan/%s.png" % (output_path, data_name)
            #     sns.pairplot(r, hue='predict', height=6, kind='scatter', diag_kind='hist')
            #     plt.savefig(save_path)
            #     plt.clf()
            #
            #     save_path = "%s/r_csv/%s.csv" % (output_path, data_name)
            #     r.to_csv(save_path, mode='w')
            #
            #     save_path = "%s/r_overlay/%s.png" % (output_path, data_name)
            #     cv_image = cv2.imread(str(image_path) + str(image))
            #     cv2.imwrite(save_path, cv_image)



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

    data = '20211225-23:35_CELL1101-1_500'
    img_data = 'CELL1101-1'
    kumar = False

    if kumar:
        mat_path = '../../output/kumar/' + data + '/mat/'
        image_path = '../../datasets/image500/' + img_data + '/'
        output_path = '../../output/filter/kumar/' + data
    else:
        mat_path = '../../output/consep/' + data + '/mat/'
        image_path = '../../datasets/image500/' + img_data + '/'
        output_path = '../../output/filter/consep/' + data

    folder(kumar, output_path)

    dbscan_filter(mat_path, output_path, image_path, kumar)
    #mat2vein(mat_path, image_path)
    end = time.time()
    print(datetime.timedelta(seconds=end-start))