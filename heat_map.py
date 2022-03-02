import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.ndimage.filters import gaussian_filter
import cv2
import io
from PIL import Image
import numpy as np
import matplotlib.pylab as plt
import scipy.io
from skimage import io
import os
from openslide import OpenSlide
from tqdm import tqdm



def myplot(x, y, s, bins=1000):
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
    heatmap = gaussian_filter(heatmap, sigma=s)

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return heatmap.T, extent


def save_heat_map(image, mat_file_value, save_path):
    # Generate data
    x = mat_file_value[:, 0]
    y = mat_file_value[:, 1]

    s = 64

    img, extent = myplot(x, y, s)

    plt.axis('off')

    image = plt.imread(image)
    print(image.shape)

    plt.imshow(image)
    plt.imshow(img, extent=extent, origin='lower', cmap=cm.jet, alpha=0.3)
    with io.BytesIO() as buff:
        plt.savefig(buff, format="png", bbox_inches='tight', pad_inches=0)
        buff.seek(0)
        cv_image = plt.imread(buff)
        buff.close()

    return cv_image


def cal_margin(level_1, crop):
    level_1_ex = 0

    while level_1 % crop != 0:
        level_1 += 1
        level_1_ex += 1

    print('level1 : ', level_1)
    print('level1_ex : ', level_1_ex)
    print('cal extend end')

    return level_1, level_1_ex


def del_y_margin(level_2, crop):
    level_2_top = int(level_2 * 0.25)
    level_2_bot = int(level_2 * 0.8)

    while level_2_top % crop != 0:
        level_2_top += 1

    while level_2_bot % crop != 0:
        level_2_bot += 1
    print('level2_top : ', level_2_top)
    print('level2_bot : ', level_2_bot)
    print('cal del y end')

    return level_2_top, level_2_bot


def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))

    return result


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


def concat_tile(im_list_2d):
    return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])


def tile_cross(level_1, level_2, image, crop, mat_path, data_name):
    y = 0
    y_for = (int(level_2 / crop) * 2) -1
    x_for = (int(level_1 / crop) * 2) -1

    img_white = Image.new('RGBA', (crop, crop), (255, 255, 255, 255))
    img_white_np = np.asarray(img_white)
    tile = [[] for _ in range(y_for)]

    file_cnt = 0


    print('start check crop')
    for i in tqdm(range(y_for)):
        x = 0
        for j in range(x_for):
            img_crop = image.crop((x, y, x + crop, y + crop))

            try:
                if file_cnt % 2 == 0:
                    mat_file_centroid, mat_file_type, mat_file_map = heat_map.load_mat(mat_path + 'img_' + file_cnt + '.mat')
                    mat_file_value = heat_map.green_filter(mat_file_centroid, mat_file_type, mat_file_map)

                    img = save_heat_map(img_crop, mat_file_value, '../output/heat_map/' + data_name + '.png')
                    im1_s = cv2.resize(img, dsize=(0, 0), fx=0.1, fy=0.1)
                    tile[i].append(im1_s)

            except:
                if j % 2 == 0 and i % 2 == 0:
                    im1_s = cv2.resize(img_crop, dsize=(0, 0), fx=0.1, fy=0.1)
                    tile[i].append(im1_s)

            x += crop // 2
            file_cnt += 1

        y += crop // 2

    return tile

class heat_map:
    def __init__(self):
        self.mat_file_name = None
        self.mat_file = None
        self.mat_file_centroid = None
        self.mat_file_type = None
        self.mat_file_value = None
        self.mat_file_map = None

    def load_mat(self, path):
        self.mat_file_name = path
        self.mat_file = scipy.io.loadmat(self.mat_file_name)

        self.mat_file_centroid = self.mat_file['inst_centroid']
        self.mat_file_type = self.mat_file['inst_type']

        self.mat_file_map = np.c_[self.mat_file_type, self.mat_file_centroid]

        return self.mat_file_centroid, self.mat_file_type, self.mat_file_map

    def green_filter(self, mat_file_centroid, mat_file_type, mat_file_map):
        self.mat_file_value = []

        for i in range(len(mat_file_type)):
            if mat_file_map[:, 0][i] == 2:
                self.mat_file_value.append(mat_file_centroid[i].tolist())
        self.mat_file_value = np.asarray(self.mat_file_value)

        return self.mat_file_value

def get_data_name(path):
    image_path = path.split('/')
    data_name = image_path[4].split('.')[0]

    return data_name

if __name__ == '__main__':
    crop = 500

    mat_path = '../output/2021-12-06_CELL1101-1_500/a_mat'
    mrxs_path = '../datasets/mrxs/CELL1101-1.mrxs'

    data_name = get_data_name(mrxs_path)

    wsi = OpenSlide(mrxs_path)

    level_dim = wsi.level_dimensions
    level_1 = level_dim[0][0]
    level_2 = level_dim[0][1]

    img = wsi.read_region((0, 0), 0, (level_1, level_2))

    level_1_x, level_1_ex = cal_margin(level_1, crop)
    level_2_top, level_2_bot = del_y_margin(level_2, crop)

    img = img.crop((0, level_2_top, level_1, level_2_bot))

    img = add_margin(img, 0, level_1_ex, 0, 0, (255, 255, 255))

    level_1 = img.size[0]
    level_2 = img.size[1]

    tile = tile_cross(level_1, level_2, img, crop, mat_path, data_name)
    del (img)
    im_tile = concat_tile(tile)
    del (tile)
    cv2.imwrite('../datasets/image' + str(crop) + '/' + data_name + '_opencv_concat_tile.jpg', im_tile)
