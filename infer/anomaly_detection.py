import pandas as pd
from sklearn.cluster import DBSCAN
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def cell_by_color(mat_file_centroid, mat_file_type):
    mat_file_map = np.c_[mat_file_type, mat_file_centroid]
    mat_file_red = []
    mat_file_value = []
    mat_file_blue = []
    # red = 1, green = 2, blue = 3, black = 0
    # red
    for i in range(len(mat_file_type)):
        if mat_file_map[:, 0][i] == 1:
            mat_file_red.append(mat_file_centroid[i].tolist())
    mat_file_red = np.asarray(mat_file_red)

    # green
    for i in range(len(mat_file_type)):
        if mat_file_map[:, 0][i] == 2:
            mat_file_value.append(mat_file_centroid[i].tolist())
    mat_file_value = np.asarray(mat_file_value)

    # blue
    for i in range(len(mat_file_type)):
        if mat_file_map[:, 0][i] == 3:
            mat_file_blue.append(mat_file_centroid[i].tolist())
    mat_file_blue = np.asarray(mat_file_blue)

    red = len(mat_file_red)
    green = len(mat_file_value)
    blue = len(mat_file_blue)

    return red, green, blue, mat_file_value, mat_file_red, mat_file_blue


def dbscan_green(mat_file_value):
    mat_file_value = pd.DataFrame(mat_file_value)
    model = DBSCAN(eps=35, min_samples=4, n_jobs=-1)
    predict = pd.DataFrame(model.fit_predict(mat_file_value))
    predict_list = predict.values.tolist()
    predict_list_1d = sum(predict_list, [])
    len_predict = len(list(set(predict_list_1d)))
    # and red > 6 and red < 34 and green < 28 and blue < 85 and blue > 9
    predict.columns = ['predict']
    # concatenate labels to df as a new column
    r = pd.concat([mat_file_value, predict], axis=1)

    return r, len_predict

def dbscan_red(mat_file_value):
    mat_file_value = pd.DataFrame(mat_file_value)
    model = DBSCAN(eps=15, min_samples=4, n_jobs=-1)
    predict = pd.DataFrame(model.fit_predict(mat_file_value))
    predict_list = predict.values.tolist()
    predict_list_1d = sum(predict_list, [])
    len_predict = len(list(set(predict_list_1d)))
    # and red > 6 and red < 34 and green < 28 and blue < 85 and blue > 9
    predict.columns = ['predict']
    # concatenate labels to df as a new column
    r = pd.concat([mat_file_value, predict], axis=1)

    return r, len_predict


def dbscan_kumar(mat_file_value):
    mat_file_value = pd.DataFrame(mat_file_value)
    model = DBSCAN(eps=35, min_samples=4, n_jobs=-1)
    predict = pd.DataFrame(model.fit_predict(mat_file_value))
    predict_list = predict.values.tolist()
    predict_list_1d = sum(predict_list, [])
    len_predict = len(list(set(predict_list_1d)))
    # and red > 6 and red < 34 and green < 28 and blue < 85 and blue > 9
    predict.columns = ['predict']
    # concatenate labels to df as a new column
    r = pd.concat([mat_file_value, predict], axis=1)

    return r, len_predict