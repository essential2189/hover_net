# OpenCV Test .py
import math
import os
import numpy as np
import cv2
from openslide import OpenSlide
from operator import itemgetter
from common.Constants import Constants
import datetime
# Color Extract with BGR => HSV
import random
# blue = np.uint8([[[255,0,0]]])
# hsv_blue = cv2.cvtColor(blue, cv2.COLOR_BGR2HSV)
# print(hsv_blue)

class ExtractVenous(object):
    def __init__(self):
        self.kernel = np.ones((5,5), np.uint8)
        self.dil_kernel = np.ones((10,10), np.uint8)
        self.dis_marker_size = 5

    def main(self):
        for file_nm in Constants.file_list:
            img = cv2.imread(Constants.data_path+file_nm, cv2.IMREAD_COLOR)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, bin_img = cv2.threshold(gray_img, 180, 255, cv2.THRESH_BINARY)
            open_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, self.kernel)
            ero_open_img = cv2.dilate(open_img, self.dil_kernel)

            dist_img = cv2.distanceTransform(ero_open_img, cv2.DIST_L2, self.dis_marker_size)
            _, dist_bin_img = cv2.threshold(dist_img, 30, 255, cv2.THRESH_BINARY)
            dist_bin_img = np.uint8(dist_bin_img)
            unknown = cv2.subtract(ero_open_img, dist_bin_img)
            # make marker
            _, marker = cv2.connectedComponents(dist_bin_img)
            marker = marker+1
            marker[unknown==255]= 0

            result = cv2.watershed(img, marker)

            img[result==1] = [0, 0, 0]
            result = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(Constants.result_path+"vein"+file_nm, result)

class ExtractBileDuct(object):
    def __init__(self):
        self.kernel = np.ones((4,4), np.uint8)
        self.dis_marker_size = 3

    def main(self):
        for file_nm in Constants.file_list:
            img = cv2.imread(Constants.data_path+file_nm, cv2.IMREAD_COLOR)
            blue = 255-img[:,:,2]
            green = img[:,:,1]
            red = 255-img[:,:,0]
            zeros = np.zeros(blue.shape)

            temp_res = np.uint8(np.dstack((red, zeros, blue)))
            gray_temp_res = cv2.cvtColor(temp_res, cv2.COLOR_BGR2GRAY)
            norm_gray_temp_res = cv2.normalize(gray_temp_res, None, 0, 255, cv2.NORM_MINMAX)
            _, bin_temp_res = cv2.threshold(norm_gray_temp_res, 100, 255, cv2.THRESH_TOZERO)
            # blur = cv2.blur(bin_temp_res, (7,7))
            open_temp_res = cv2.morphologyEx(bin_temp_res, cv2.MORPH_OPEN, self.kernel)
            edge_temp_res = cv2.Canny(open_temp_res, 200, 255)

            dist_bin_img = np.uint8(bin_temp_res)
            unknown = cv2.subtract(bin_temp_res, dist_bin_img)
            # make marker
            _, marker = cv2.connectedComponents(dist_bin_img)
            marker = marker+1
            marker[unknown==255]= 0

            result = cv2.watershed(img, marker)

            img[result==1] = [0, 0, 0]
            cv2.imwrite(Constants.result_path+"bileduct_"+file_nm, bin_temp_res)
            cv2.imwrite(Constants.test_path+"bileduct_edge"+file_nm, edge_temp_res)
            cv2.imwrite(Constants.test_path+"bileduct_seg"+file_nm, result)
            # cv2.imwrite(Constants.test_path+"bileduct_blur"+file_nm, blur)



class ExtractTriad(object):
    def __init__(self):
        self.kernel = np.ones((5,5), np.uint8)

    def main(self):
        for file_nm in Constants.file_list:
            img = cv2.imread(Constants.data_path+file_nm, cv2.IMREAD_COLOR)
            blue = 255-img[:,:,0]
            red = img[:,:,2]
            zeros = np.zeros(blue.shape)

            temp_res = np.uint8(np.dstack((red, zeros, blue/2)))
            gray_temp_res = cv2.cvtColor(temp_res, cv2.COLOR_BGR2GRAY)
            norm_gray_temp_res = cv2.normalize(gray_temp_res, None, 0, 255, cv2.NORM_MINMAX)
            _, bin_temp_res = cv2.threshold(norm_gray_temp_res, 140, 255, cv2.THRESH_TOZERO)

            open_temp_res = cv2.morphologyEx(bin_temp_res, cv2.MORPH_OPEN, self.kernel)

            # cv2.imwrite(Constants.result_path+"bin"+file_nm, bin_temp_res)
            cv2.imwrite(Constants.result_path+"triad"+file_nm, open_temp_res)


class mrxsExtractor(object):
    def __init__(self):
        data_path = 'data/mrx3/liver.mrxs'
        self.data_path = data_path
        self.low_dim = 6
        self.top_n_label = 2
        self.high_dim = 5
        self.export_res = [1000,1000]
        self.wsi = OpenSlide(data_path)
        self.mrxs_spec= self.wsi.level_dimensions
        print(self.mrxs_spec)

        self.min_x = self.mrxs_spec[self.low_dim][0]
        self.max_x = 0
        self.min_y = self.mrxs_spec[self.low_dim][1]
        self.max_y = 0

    def main(self):
        labeled_map = self.low_dim_extract()
        extracted_img = self.high_dim_extract(labeled_map)
        self.img_extract_tile()
        pass


    def img_extract_tile(self):
        w0 = self.mrxs_spec[0][0]/self.mrxs_spec[self.low_dim][0]
        print(w0)
        init_x = math.floor(self.min_x*w0)
        init_y = math.floor(self.min_y*w0)
        w_high = self.mrxs_spec[self.high_dim][0]/self.mrxs_spec[self.low_dim][0]

        len_x = math.ceil((self.max_x-self.min_x)*w_high/self.export_res[0])
        len_y = math.ceil((self.max_y-self.min_y)*w_high/self.export_res[1])

        print(init_x,init_y, len_x, len_y)
        for j in range(len_y):
            for i in range(len_x):
                x=int(init_x+self.export_res[0]*i*w0/w_high)
                y=int(init_y+self.export_res[1]*j*w0/w_high)
                img = np.uint8(self.wsi.read_region((x,y), self.high_dim, self.export_res))

                cv2.imwrite("data/test/high_test"+str(i)+"_"+str(j)+".png", img)


    #     print(x,y, i)
        #
        #     if x>=max_x:
        #         break
        #     img = np.uint8(self.wsi.read_region((x,y), self.high_dim, self.export_res))
        #     x+=self.export_res[0]
        #     y+=self.export_res[1]
        #     cv2.imwrite("data/test/high_test"+str(i)+".png", img)
        #     i+=1



    def img_extract_window(self):
        pass
    def high_dim_extract(self, labeled_img:np.uint8):
        # result = np.uint8(self.wsi.read_region((0,0), 0, self.mrxs_spec[0]))
        # result[labeled_img==0]=0
        # return result
        pass

    def low_dim_extract(self):
        img = np.uint8(self.wsi.read_region((0,0), self.low_dim, self.mrxs_spec[self.low_dim]))
        cvt_img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

        cvt_img[cvt_img==0]=255

        _, cvt_img = cv2.threshold(cvt_img, 230, 255, cv2.THRESH_BINARY_INV)
        cv2.imwrite("data/test/thresh.png", cvt_img)

        a,b = cv2.connectedComponents(cvt_img)

        label_dict = dict()
        for i in range(a):
            if i==0:
                continue
            label_dict[i] = len(b[b==i])
        sort_label_list = sorted(label_dict.items(), key=lambda item:item[1], reverse=True)

        result = np.zeros(b.shape)
        for ins in sort_label_list[:self.top_n_label]:
            result[b==ins[0]]=255

        # test = cv2.resize(result, self.mrxs_spec[0])
        for y, ins in enumerate(result):
            for x, val in enumerate(ins):
                if val!=0:
                    if self.min_x>=x:
                        self.min_x = x
                    if self.max_x<=x:
                        self.max_x = x
                    if self.min_y>=y:
                        self.min_y = y
                    if self.max_y<=y:
                        self.max_y = y
        print(self.min_x, self.max_x, self.min_y, self.max_y)
        roi_result = result[self.min_y:self.max_y,self.min_x:self.max_x]
        cv2.imwrite("data/test/roi_result.png", roi_result)
        # return test

if __name__ == '__main__':
    mrxs = mrxsExtractor()
    mrxs.main()
    # # Extract Triad
    # tri = ExtractTriad()
    # tri.main()
    #
    # Extract Venous
    # ven = ExtractVenous()
    # ven.main()
    # #
    # # Extract Bile Duct
    # bile = ExtractBileDuct()
    # bile.main()

    # print(str(datetime.date.today().month)+str(datetime.date.today() .day))
    # img = cv2.imread("data/test.png", cv2.IMREAD_COLOR)
    # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ret, bin_img = cv2.threshold(gray_img, 100,255,0)
    # contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # cv2.imwrite("data/result.png",np.uint8(contours))

