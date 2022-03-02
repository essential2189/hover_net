import os
import pandas as pd
import time
import datetime
import cv2

def read_csv(csv_path):
    csv = pd.read_csv(csv_path, header=None)

    return csv



def data_split(img_path, csv_path):
    csv = read_csv(csv_path)


    for name, label in zip(csv[0], csv[1]):
        print(name, label)
        if label == 'SSA':
            img = cv2.imread(img_path + name)
            cv2.imwrite('/home/sjwang/mhist/SSA/' + name, img)
        elif label == 'HP':
            img = cv2.imread(img_path + name)
            cv2.imwrite('/home/sjwang/mhist/HP/' + name, img)
        del img


if __name__ == '__main__':
    start = time.time()
    csv_path = '/home/sjwang/mhist/annotations.csv'
    img_path = '/home/sjwang/mhist/images/'
    data_split(img_path, csv_path)
    end = time.time()
    print(datetime.timedelta(seconds=end-start))



