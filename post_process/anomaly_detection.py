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


def dbscan_green(mat_file_value):
    mat_file_value = pd.DataFrame(mat_file_value)
    model = DBSCAN(eps=35, min_samples=5, n_jobs=-1)
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
    model = DBSCAN(eps=18, min_samples=3, n_jobs=-1)
    predict = pd.DataFrame(model.fit_predict(mat_file_value))
    predict_list = predict.values.tolist()
    predict_list_1d = sum(predict_list, [])
    len_predict = len(list(set(predict_list_1d)))
    # and red > 6 and red < 34 and green < 28 and blue < 85 and blue > 9
    predict.columns = ['predict']
    # concatenate labels to df as a new column
    r = pd.concat([mat_file_value, predict], axis=1)

    return r, len_predict

def dbscan_green_red(mat_file_value):
    mat_file_value = pd.DataFrame(mat_file_value)
    model = DBSCAN(eps=30, min_samples=6, n_jobs=-1)
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
    model = DBSCAN(eps=45, min_samples=6, n_jobs=-1)
    predict = pd.DataFrame(model.fit_predict(mat_file_value))
    predict_list = predict.values.tolist()
    predict_list_1d = sum(predict_list, [])
    len_predict = len(list(set(predict_list_1d)))
    # and red > 6 and red < 34 and green < 28 and blue < 85 and blue > 9
    predict.columns = ['predict']
    # concatenate labels to df as a new column
    r = pd.concat([mat_file_value, predict], axis=1)

    return r, len_predict



def dbscan_green_red_nonePD(mat_file_value, eps, minsample):
    model = DBSCAN(eps=eps, min_samples=minsample)
    predict = model.fit_predict(mat_file_value)
    # predict_list = predict.tolist()
    len_predict = set(predict)
    # predict_dict = dict(enumerate(predict_list))
    # predict['predict'] = predict_dict.values()
    # r = pd.concat([mat_file_value, predict], axis=1)

    return len(len_predict)


def dbscan_kumar_nonePD(mat_file_value):
    model = DBSCAN(eps=30, min_samples=6, n_jobs=-1)
    predict = model.fit_predict(mat_file_value)
    # predict_list = predict.tolist()
    len_predict = set(predict)
    # predict_dict = dict(enumerate(predict_list))
    # predict['predict'] = predict_dict.values()
    # r = pd.concat([mat_file_value, predict], axis=1)

    return len(len_predict)

def kmean_consep(X):
    Kmean = KMeans(n_clusters=20)
    predict = Kmean.fit(X)

    return predict