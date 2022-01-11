import os
import pandas as pd



def read_csv(csv_path):
    return pd.read_csv(csv_path, index_col=0).fillna(-1).astype('int')



def check_ann(cnt_list, csv_path):
    ann_csv = read_csv(csv_path)
    id_list = ann_csv.index.to_list()
    detect = False

    name_list = []
    for file_name in cnt_list:
        name = file_name.split('.')[0]
        name = name.split('_')[1]
        name_list.append(int(name))

    for i in ann_csv.index:
        for j in ann_csv.loc[i]:
            if j != -1:
                if j in name_list:
                    name_list.remove(j)
                    detect = True

        if detect:
            id_list.remove(i)
            detect = False

    return name_list, id_list



