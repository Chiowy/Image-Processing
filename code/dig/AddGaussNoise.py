import cv2
import numpy as np


def add_gaussian_noise(image, mean=0, var=0.001):
    """
    添加高斯噪声
    :param image:原始图像
    :param mean : 均值
    :param var : 方差,越大，噪声越大
    :return image_after_add_gaussian_noise
    """
    image = np.array(image / 255, dtype=float)  # 将原始图像的像素值进行归一化，除以255使得像素值在0-1之间
    noise = np.random.normal(mean, var ** 0.5, image.shape)  # 创建一个均值为mean，方差为var呈高斯分布的图像矩阵
    out = image + noise  # 将噪声和原始图像进行相加得到加噪后的图像
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)  # clip函数将元素的大小限制在了low_clip和1之间了，小于的用low_clip代替，大于1的用1代替
    out = np.uint8(out * 255)  # 解除归一化，乘以255将加噪后的图像的像素值恢复

    return out


if __name__ == '__main__':
    img = cv2.imread('/code/dig/Anastasia.jpg')
    img_after_added_noise = add_gaussian_noise(img)
    cv2.imshow('noise', img_after_added_noise)
    cv2.waitKey(0)
