import os
import sys

import torch
import torch.utils.data as data

import numpy as np
from PIL import Image
import glob
import random
import cv2

random.seed(1143)

# folder_path = "sum"  # 文件夹路径
#
# # 遍历文件夹及其子文件夹中的所有文件
# for root, dirs, files in os.walk(folder_path):
#     for file in files:
#         file_path = os.path.join(root, file)
#         print(file_path)


def LOL_train_dataset(lowlight_images_path):
    file_paths_and_names = []

    # 遍历文件夹
    for dirpath, dirnames, filenames in os.walk(lowlight_images_path):
        # 遍历文件
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            file_paths_and_names.append(file_path)

    random.shuffle(file_paths_and_names)

    return file_paths_and_names


def populate_train_list(lowlight_images_path):
    image_list_lowlight = glob.glob(lowlight_images_path + "*.png")

    train_list = image_list_lowlight

    random.shuffle(train_list)

    return train_list


class lowlight_loader(data.Dataset):

    def __init__(self, lowlight_images_path):
        self.train_list = LOL_train_dataset(lowlight_images_path)
        self.size = 128
        print("Total dataset examples:", len(self.train_list))
        if len(self.train_list) > 1000:
            self.train_list = random.sample(self.train_list, 200)

        self.data_list = self.train_list
        print("Total training examples:", len(self.data_list))



    def __getitem__(self, index):
        data_lowlight_path = self.data_list[index]

        data_lowlight = Image.open(data_lowlight_path)

        # 图像重定义尺寸为（self.size, self.size）, 重定义尺寸方法为高标准， Image.ANITALIAS，
        data_lowlight = data_lowlight.resize((self.size, self.size), Image.ANTIALIAS)

        # 将图像像素数据范围归一化到（0,1）之间，pixel/255.0
        data_lowlight = (np.asarray(data_lowlight) / 255.0)

        # 将数据格式转化为torch中的tensor数据形式
        data_lowlight = torch.from_numpy(data_lowlight).float()

        # torch.tensor.permute(*dims) -> Tensor 交换图像的维度，将列顺序转变为（2， 0， 1）
        return data_lowlight.permute(2, 0, 1)

    def __len__(self):
        return len(self.data_list)


if __name__ == '__main__':
    lowlight_images_path = "data/train_data/LOLdataset/our485/low/"
    # 获取文件路径和名称列表
    files = LOL_train_dataset(lowlight_images_path)
    print("共有图像{}张".format(len(files)))

    # # 打印文件路径和名称列表
    # for file in files:
    #     print(file)
    low_light_dataset = lowlight_loader(lowlight_images_path)





