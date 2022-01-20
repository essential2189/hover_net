import os

data = '20220113-21:04_1102-1'
img_data = '1102-1'
mat_path = '../../output/pannuke/' + data + '/mat/'
image_path = '../../datasets/image500/' + img_data + '/'

name_list = []
for file in sorted(os.listdir(mat_path)):
    name = file.split('.')[0]
    name = name.split('_')[1]
    name_list.append(int(name))

for i in range(len(os.listdir(image_path))):
    if i not in name_list:
        print(i)

print(len(os.listdir(image_path)), len(os.listdir(mat_path)))