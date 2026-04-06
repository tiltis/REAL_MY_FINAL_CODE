import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# import pytorch_colors as colors
import numpy as np
import torchvision.transforms as transforms
import cv2 as cv
import os
from thop import profile


class lle_net(nn.Module):

    def __init__(self):
        super(lle_net, self).__init__()

        # 设置迭代次数
        self.ite_number = 1
        self.output_dim = self.ite_number * 3

        # 构造3x3卷积等价并列模型， 输出结果为(16， 128， 128)
        # 输入通道为3， 输出通道数为16， 卷积核为3x3, 步长为1， padding为1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1, bias=True)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True)
        self.avgpool1 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(16)

        # 构造第2个等价3x3卷积，输入为（16， 64， 64）， 输出为（32， 64， 64）
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=1, bias=True)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv6 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)
        self.avgpool2 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(32)

        # 构造第三个等价3x3卷积，输入为（32， 32， 32）， 输出为(64, 32, 32)
        self.conv7 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, bias=True)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv8 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv9 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.avgpool3 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(64)

        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.gelu1 = nn.GELU()
        self.sigmoid = nn.Sigmoid()

        self.bool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64, self.output_dim)

    def forward(self, x):
        def LLE_LUT(origin_img, t, b):
            batch_size, c, h, w = origin_img.shape
            # 对t, b升维（batch_size, 3, 256, 1）
            t = t.unsqueeze(2).unsqueeze(2).expand(batch_size, 3, 256, 1).requires_grad_(True).cuda()
            b = b.unsqueeze(2).unsqueeze(2).expand(batch_size, 3, 256, 1).requires_grad_(True).cuda()

            x_init = (torch.arange(256).reshape(-1, 1) / 255.0).requires_grad_(True).cuda()
            x_init = torch.unsqueeze(torch.unsqueeze(x_init, 0), 0).expand(batch_size, 3, 256, 1)

            # 根据变换规则进行映射
            lookup_tables = ((x_init - t) ** 3 + t ** 3) * (1 - b) / ((1 - t) ** 3 + t ** 3) + b * x_init
            # 变为 0~255之间的像素值
            lookup_tables = torch.round(lookup_tables * 255)

            origin_img = origin_img * 255.0
            origin_img = origin_img.view(batch_size, c, -1, 1).to(dtype=torch.int64)  # 将输入向量变为（b, c, h*w, 1）
            enhance_image = torch.gather(lookup_tables, dim=2, index=origin_img)
            enhance_image = enhance_image / 255.0
            enhance_image = enhance_image.view(batch_size, c, h, w)

            return enhance_image

        def generate_lookup_table(batch_size, t, b):
            '''
            本函数为生成对应查找表的函数。
            :param batch_size:pytorch 中 batch_size 值。
            :param t: 维度为（batch_size, 3）,其中3 为对应的3通道
            :param b: 维度为（batch_size, 3）,其中3 为对应的3通道
            :return: 依据规则生成的查找表， 维度信息为（batch_size, 3, 256)
            '''

            # print("-------------t is---------")
            # print(t)
            # print("-------------b is---------")
            # print(b)

            # 对t, b升维（batch_size, 3, 256, 1）
            t = t.unsqueeze(2).unsqueeze(2).expand(batch_size, 3, 256, 1).requires_grad_(True).cuda()
            b = b.unsqueeze(2).unsqueeze(2).expand(batch_size, 3, 256, 1).requires_grad_(True).cuda()

            x_init = (torch.arange(256).reshape(-1, 1) / 255.0).requires_grad_(True).cuda()
            x_init = torch.unsqueeze(torch.unsqueeze(x_init, 0), 0).expand(batch_size, 3, 256, 1)

            # 根据变换规则进行映射
            lookup_tables = ((x_init - t) ** 3 + t ** 3) * (1 - b) / ((1 - t) ** 3 + t ** 3) + b * x_init
            # 变为 0~255之间的像素值
            lookup_tables = torch.round(lookup_tables * 255)
            lookup_tables = lookup_tables

            # print("==================lookup_table=================")
            # print(lookup_tables)
            # print("lookup_tables.shape", lookup_tables.shape)

            return lookup_tables

        def LLE_LUT_formula(images, lookup_table):
            '''
            本函数为将输入的图像输出为映射后的图像
            :param images: 输入的待映射图像，四维向量（batch_size, c, h, w）
            :param lookup_table: 查找表映射，（batch_size, 3, 256, 1）
            :return: 依据查找表的映射后图像
            '''

            # 由于传进来的都是归一化图像，因此再变换到0~255区间，维度为（batch_size, h, w, c）
            b, c, h, w = images.shape
            images_mem = images * 255.0
            images_mem = images_mem.view(b, c, -1, 1).to(dtype=torch.int64)  # 将输入向量变为（b, c, h*w, 1）
            imgdst = torch.gather(lookup_table, dim=2, index=images_mem)
            enhance_image = imgdst / 255.0
            enhance_image = enhance_image.view(b, c, h, w)

            # print("===========original_image is ==============  ")
            # print(images * 255)
            # print("enhance_image shape is ", enhance_image.shape)
            # print("===========enhance_image is ==============  ")
            # print(enhance_image * 255)
            # print("enhance_image shape is ", enhance_image.shape)

            return enhance_image

        # 构建各个并列分支


        # branch11 = self.bn1(self.conv2(x))
        # branch12 = self.bn2(self.conv3(self.bn1(self.conv2(x))))
        # branch13 = self.bn2(self.avgpool1(self.bn1(self.conv2(x))))
        # branch14 = self.bn1(self.conv1(x))
        # branch1 = branch11 + branch12 + branch13 + branch14

        # 将并行分支整体替换为3*3卷积
        x1 = self.conv1(x)

        x1 = self.avgpool(x1)
        x1 = self.gelu1(x1)
        x1 = self.sigmoid(x1)

        # branch21 = self.bn3(self.conv4(x3))
        # branch22 = self.bn4(self.conv6(self.bn3(self.conv4(x3))))
        # branch23 = self.bn4(self.avgpool2(self.bn3(self.conv4(x3))))
        # branch24 = self.bn3(self.conv5(x3))
        # branch2 = branch21 + branch22 + branch23 + branch24

        # 将并行分支整体替换为3*3卷积
        x1 = self.conv4(x1)

        x1 = self.avgpool(x1)
        x1 = self.gelu1(x1)
        x1 = self.sigmoid(x1)

        # branch31 = self.bn5(self.conv7(x6))
        # branch32 = self.bn6(self.conv9(self.bn5(self.conv7(x6))))
        # branch33 = self.bn6(self.avgpool3(self.bn5(self.conv7(x6))))
        # branch34 = self.bn5(self.conv8(x6))
        # branch3 = branch31 + branch32 + branch33 + branch34

        # 将并行分支整体替换为3*3卷积
        x1 = self.conv7(x1)

        x1 = self.avgpool(x1)
        x1 = self.gelu1(x1)
        x1 = self.sigmoid(x1)

        x1 = self.bool(x1)
        x1 = self.flatten(x1)
        x1 = self.fc(x1)
        x1 = self.sigmoid(x1)

        def lut_init(x_origin, t, b):
            x_enhance = (1 - b[:, :, None, None]) * (
                    (x_origin - t[:, :, None, None]) ** 3) / (
                                (1 - t[:, :, None, None]) ** 3 +
                                t[:, :, None, None] ** 3) + b[:, :, None, None] * x_origin + (
                                t[:, :, None, None] ** 3) * (1 - b[:, :, None, None]) / (
                                (1 - t[:, :, None, None]) ** 3 +
                                t[:, :, None, None] ** 3)

            return x_enhance

        def gamma_trans(x_origin, t, b):
            alpha_multi_gamma = 1/(torch.pow(t[:, :, None, None], b[:, :, None, None]))
            x_enhance = alpha_multi_gamma * ((t[:, :, None, None] * x_origin) ** b[:, :, None, None])

            return x_enhance

        def gamma_trans_simply(x_origin, t, b):

            x_enhance = torch.pow(x_origin, t[:, :, None, None]) + b[:, :, None, None]

            return x_enhance

        def gamma_simply(x_origin, t):

            x_enhance = torch.pow(x_origin, t[:, :, None, None])

            return x_enhance

        # t1 = torch.split(x1, 3, dim=1)
        # t1, b1, t2, b2, t3, b3, t4, b4, t5, b5, t6, b6 = torch.split(x12, 3, dim=1)

        # enhance_image_1 = lut_init(x, t1, b1)
        # enhance_image_2 = lut_init(enhance_image_1, t2, b2)
        # enhance_image_3 = lut_init(enhance_image_2, t3, b3)
        # enhance_image_4 = lut_init(enhance_image_3, t4, b4)

        enhance_image_1 = gamma_simply(x, x1)
        # enhance_image_2 = gamma_trans_simply(enhance_image_1, t2, b2)
        # enhance_image_3 = gamma_trans(enhance_image_2, t3, b3)
        # enhance_image_4 = gamma_trans(enhance_image_3, t4, b4)


        # enhance_image_5 = lut_init(enhance_image_4, t5, b5)
        # enhance_image_6 = lut_init(enhance_image_5, t6, b6)

        # t1, b1, t2, b2, t3, b3, t4, b4 = torch.split(x12, 3, dim=1)
        # enhance_image_1 = LLE_LUT(x, t1, b1)
        # enhance_image_2 = LLE_LUT(enhance_image_1, t2, b2)
        # enhance_image_3 = LLE_LUT(enhance_image_2, t3, b3)
        # enhance_image_4 = LLE_LUT(enhance_image_3, t4, b4)

        # batch_size = x.shape[0]
        # lookup_table_1 = generate_lookup_table(batch_size, t1, b1)
        # enhance_image_1 = LLE_LUT_formula(x, lookup_table_1)
        #
        # lookup_table_2 = generate_lookup_table(batch_size, t2, b2)
        # enhance_image_2 = LLE_LUT_formula(enhance_image_1, lookup_table_2)
        #
        # lookup_table_3 = generate_lookup_table(batch_size, t3, b3)
        # enhance_image_3 = LLE_LUT_formula(enhance_image_2, lookup_table_3)
        #
        # lookup_table_4 = generate_lookup_table(batch_size, t4, b4)
        # enhance_image_4 = LLE_LUT_formula(enhance_image_3, lookup_table_4)

        return enhance_image_1, x1   # enhance_image_3, enhance_image_4,


if __name__ == "__main__":
    # 测试模型是否运行正常
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    model = lle_net().cuda()
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 打印参数数量
    print(f"The network has {num_params} trainable parameters.")
    print(model)
    images = torch.randint(0, 256, (1, 3, 1200, 900), dtype=torch.float32, requires_grad=True)
    print("==============images=========")





    # print(images)
    # input_tensor = torch.ones((4, 3, 128, 128))
    input = images / 255.0
    input = input.cuda()
    le1, params = model(input)
    print("output.shape is ", le1.shape)

    flops, params = profile(model, inputs=(input,))
    print(f"Estimated FLOPs: {flops}")
    print(f"Number of parameters: {params}")
