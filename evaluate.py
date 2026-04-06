import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio
import pyiqa


def calculate_metrics(image1_path, image2_path):
    # 读取图像
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    # PSNR
    psnr_value = peak_signal_noise_ratio(img1, img2)

    # LPIPS, SSIM
    lpips_metric = pyiqa.create_metric('lpips').cuda()
    ssim_metric = pyiqa.create_metric('ssim').cuda()

    lpips_value = lpips_metric(image1_path, image2_path)
    ssim_value = ssim_metric(image1_path, image2_path)

    # MAE
    mae_value = mean_squared_error(img1, img2)

    return psnr_value, lpips_value, ssim_value, mae_value

def calculate_pi_niqe(image_path):

    # create metric function
    niqe_metric = pyiqa.create_metric('niqe').cuda()
    pi_metric = pyiqa.create_metric('pi').cuda()

    niqe_score = niqe_metric(image_path)
    # print(f'NIQE score: {niqe_score:.4f}')
    pi_score = pi_metric(image_path)
    # print(f'NIQE score_2: {niqe_score:.4f}')

    return pi_score, niqe_score

def evaluate_images(folder1, folder2):
    # 获取两个文件夹中的图像文件列表
    images1 = os.listdir(folder1)
    images2 = os.listdir(folder2)

    # 确保图像文件名称相等
    assert images1 == images2, "文件夹内的图像文件不匹配"

    # 初始化指标总和
    psnr_sum = 0
    lpips_sum = 0
    ssim_sum = 0
    mae_sum = 0
    pi_sum = 0
    niqe_sum = 0

    # 循环计算每对图像的指标
    for image_name in images1:
        image1_path = os.path.join(folder1, image_name)
        image2_path = os.path.join(folder2, image_name)

        psnr, lpips, ssim, mae = calculate_metrics(image1_path, image2_path)
        pi, niqe = calculate_pi_niqe(image2_path)

        psnr_sum += psnr
        lpips_sum += lpips
        ssim_sum += ssim
        mae_sum += mae
        pi_sum += pi
        niqe_sum += niqe

    # 计算平均指标
    num_images = len(images1)
    psnr_avg = psnr_sum / num_images
    lpips_avg = lpips_sum /num_images
    ssim_avg = ssim_sum / num_images
    mae_avg = mae_sum / num_images
    pi_avg = pi_sum / num_images
    niqe_avg = niqe_sum / num_images

    print(f'Average PSNR: {psnr_avg:.4f}')
    print(f'Average LPIPS: {lpips_avg:.4f}')
    print(f'Average SSIM: ', ssim_avg)
    print(f'Average MAE: {mae_avg:.4f}')
    print(f'Average Perceptual Index (PI): {pi_avg:.4f}')
    print(f'Average NIQE: {niqe_avg:.4f}')


if __name__ == '__main__':
    # 示例使用
    folder1 =r'data/test_data/LIME/' # 替换为file1的路径
    folder2 =r'data/result/con_ite_4(E-0.4)/LIME/'  # 替换为file2的路径

    evaluate_images(folder1, folder2)

