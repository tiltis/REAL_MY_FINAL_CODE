import torch
# import torch.nn as nn
import torchvision
import torchvision.utils
# import torch.backends.cudnn as cudnn
import torch.optim
import os
# import sys
# import argparse
# import time
# import dataloader
import model_ite_num_2
import numpy as np
# from torchvision import transforms
from PIL import Image
import glob
import time


def lowlight(image_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    data_lowlight = Image.open(image_path)

    data_lowlight = (np.asarray(data_lowlight) / 255.0)

    data_lowlight = torch.from_numpy(data_lowlight).float()
    data_lowlight = data_lowlight.permute(2, 0, 1)
    data_lowlight = data_lowlight.cuda().unsqueeze(0)

    DCE_net = model_ite_num_2.lle_net().cuda()
    DCE_net.load_state_dict(torch.load('snapshots/Epoch39.pth'))
    # start = time.time()
    # _, _, _, enhanced_image = DCE_net(data_lowlight)

    ite_1_img, enhanced_image, paras = DCE_net(data_lowlight)
    # print("enhacen parameters are", paras)
    # ite_1_img, ite_2_img, ite_3_img, enhanced_image_8, paras = DCE_net(data_lowlight)

    # end_time = (time.time() - start)
    # print(end_time)
    image_path = image_path.replace('test_data', 'result')
    result_path = image_path
    if not os.path.exists(image_path.replace('/' + image_path.split("/")[-1], '')):
        os.makedirs(image_path.replace('/' + image_path.split("/")[-1], ''))

    # torchvision.utils.save_image(ite_1_img, result_path)
    # torchvision.utils.save_image(ite_2_img, result_path)
    # torchvision.utils.save_image(ite_3_img, result_path)
    torchvision.utils.save_image(enhanced_image, result_path)


if __name__ == '__main__':
    start_time = time.time()

    # test_images
    with torch.no_grad():
        filePath = r'data/test_data/'

        file_list = os.listdir(filePath)

        for file_name in file_list:
            test_list = glob.glob(filePath + file_name + "/*")
            for image in test_list:
                # image = image
                # print(image)
                lowlight(image)

        # 此段代码为单张测试用
        # filePath = r'data/test_data/LIME/4.bmp'
        # lowlight(filePath)

    end_time = time.time() - start_time
    print("running time is", end_time)

