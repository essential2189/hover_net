import pandas as pd
from sklearn.cluster import DBSCAN
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

def cell_by_color(mat_file_centroid, mat_file_type):
    mat_file_map = np.c_[mat_file_type, mat_file_centroid]

    # red = 1, green = 2, blue = 3, black = 0

    # red
    mat_file_red = mat_file_map[mat_file_map[:, 0][:] == 1]
    mat_file_red = np.asarray(mat_file_red)

    # green
    mat_file_value = mat_file_map[mat_file_map[:, 0][:] == 2]
    mat_file_value = np.asarray(mat_file_value)

    # blue
    mat_file_blue = mat_file_map[mat_file_map[:, 0][:] == 3]
    mat_file_blue = np.asarray(mat_file_blue)

    mat_file_y = mat_file_map[mat_file_map[:, 0][:] == 4]
    mat_file_y = np.asarray(mat_file_y)


    return mat_file_value, mat_file_red, mat_file_blue, mat_file_y


def dbscan_nonePD(mat_file_value, eps, minsample):
    model = DBSCAN(eps=eps, min_samples=minsample)
    predict = model.fit_predict(mat_file_value)
    # predict_list = predict.tolist()
    len_predict = set(predict)
    # predict_dict = dict(enumerate(predict_list))
    # predict['predict'] = predict_dict.values()
    # r = pd.concat([mat_file_value, predict], axis=1)

    return len(len_predict)

