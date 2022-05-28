import cv2
import numpy as np


def add_salt_noise(img: np.array, snr: float):
    """
    add salt noise
    :param img: input image
    :param snr: 信噪比
    :return: img after added noise
    """
    img_after_added_noise = img.copy()
    sum_of_pixel = img.size  # 获取图像的大小
    num_of_noise = sum_of_pixel * snr  # 对img.size * snr 个像素点加噪声
    for i in range(int(num_of_noise)):
        x_i = np.random.randint(0, img.shape[0])
        x_j = np.random.randint(0, img.shape[1])
        img_after_added_noise[x_i][x_j] = 0

    return img_after_added_noise


if __name__ == '__main__':
    img = cv2.imread('E:\\CODE\\Image-Processing\\images\\Anastasia.jpg')
    img_after_added_noise = add_salt_noise(img, 0.2)
    cv2.imshow('noise', img_after_added_noise)
    cv2.waitKey(0)
