import cv2
import numpy as np

from code.dig import AddSaltNoise
from code.dig import AddGaussNoise

"""part a 调入并显示原始图像"""
img = cv2.imread('Anastasia.jpg')  # 调入原始图像
cv2.imshow('img', img)
cv2.waitKey(0)

"""part b 在图像上加入Gaussian噪声、椒盐噪声"""
"gaussian noisy"
gaussian_noise_img_1 = AddGaussNoise.add_gaussian_noise(img, var=0.001)
gaussian_noise_img_2 = AddGaussNoise.add_gaussian_noise(img, var=0.005)
gaussian_noise_img_3 = AddGaussNoise.add_gaussian_noise(img, var=0.01)
gaussian_noise_img = np.asarray([gaussian_noise_img_1, gaussian_noise_img_2, gaussian_noise_img_3])
"salt noisy"
salt_noise_img1 = AddSaltNoise.add_salt_noise(img, 0.001)
salt_noise_img2 = AddSaltNoise.add_salt_noise(img, 0.005)
salt_noise_img3 = AddSaltNoise.add_salt_noise(img, 0.01)
salt_noise_img = np.asarray([salt_noise_img1, salt_noise_img2, salt_noise_img3])

"""part c 利用预定义函数产生均值滤波器"""
img_medianBlur_3 = cv2.medianBlur(salt_noise_img[0], 3)  # 3*3 的中值滤波器
img_Blur_3 = cv2.blur(salt_noise_img[0], (3, 3))  # 3*3 的均值滤波器
"""part d 分别用均值滤波器以及中值滤波器，对加入不同噪声的图像进行处理"""
"d1 采用3x3模板，分别用均值滤波器以及中值滤波器，对加入不同水平噪声的图像进行处理"
"d1_1 噪声水平1"
img_Blur_1_3 = cv2.blur(salt_noise_img1, (3, 3))  # 均值滤波
img_medianBlur_1_3 = cv2.medianBlur(salt_noise_img1, 3)  # 中值滤波
d1_11 = np.hstack((salt_noise_img1, img_Blur_1_3, img_medianBlur_1_3))
cv2.imshow('d1_1', d1_11)
cv2.imwrite('d1_1.png', d1_11)
cv2.waitKey(0)
"d1_2 噪声水平2"
img_Blur_2_3 = cv2.blur(salt_noise_img2, (3, 3))  # 均值滤波
img_medianBlur_2_3 = cv2.medianBlur(salt_noise_img2, 3)  # 中值滤波
d1_12 = np.hstack((salt_noise_img2, img_Blur_2_3, img_medianBlur_2_3))
cv2.imshow('d1_2', d1_12)
cv2.imwrite('d1_2.png', d1_12)
cv2.waitKey(0)
"d1_2 噪声水平3"
img_Blur_3_3 = cv2.blur(salt_noise_img3, (3, 3))  # 均值滤波
img_medianBlur_3_3 = cv2.medianBlur(salt_noise_img3, 3)  # 中值滤波
d1_13 = np.hstack((salt_noise_img3, img_Blur_3_3, img_medianBlur_3_3))
cv2.imshow('d1_3', d1_13)
cv2.imwrite('d1_3.png', d1_13)
cv2.waitKey(0)

"d2 采用5x5模板，分别用均值滤波器以及中值滤波器，对加入不同水平噪声的图像进行处理"
"d2_1 噪声水平1"
img_Blur_1_5 = cv2.blur(salt_noise_img1, (5, 5))  # 均值滤波
img_medianBlur_1_5 = cv2.medianBlur(salt_noise_img1, 5)  # 中值滤波
d2_1 = np.hstack((salt_noise_img1, img_Blur_1_5, img_medianBlur_1_5))
cv2.imshow('d2_1', d2_1)
cv2.imwrite('d2_1.png', d2_1)
cv2.waitKey(0)
"d2_1 噪声水平2"
img_Blur_2_5 = cv2.blur(salt_noise_img2, (5, 5))  # 均值滤波
img_medianBlur_2_5 = cv2.medianBlur(salt_noise_img2, 5)  # 中值滤波
d2_2 = np.hstack((salt_noise_img2, img_Blur_2_5, img_medianBlur_2_5))
cv2.imshow('d2_2', d2_2)
cv2.imwrite('d2_2.png', d2_2)
cv2.waitKey(0)
"d2_3 噪声水平3"
img_Blur_3_5 = cv2.blur(salt_noise_img3, (5, 5))  # 均值滤波
img_medianBlur_3_5 = cv2.medianBlur(salt_noise_img3, 5)  # 中值滤波
d2_3 = np.hstack((salt_noise_img3, img_Blur_3_5, img_medianBlur_3_5))
cv2.imshow('d2_3', d2_3)
cv2.imwrite('d2_3.png', d2_3)
cv2.waitKey(0)

"d3 采用7x7模板，分别用均值滤波器以及中值滤波器，对加入不同水平噪声的图像进行处理"
"d3_1 噪声水平1"
img_Blur_1_7 = cv2.blur(salt_noise_img1, (7, 7))  # 均值滤波
img_medianBlur_1_7 = cv2.medianBlur(salt_noise_img1, 7)  # 中值滤波
d3_1 = np.hstack((salt_noise_img1, img_Blur_1_7, img_medianBlur_1_7))
cv2.imshow('d3_1', d3_1)
cv2.imwrite('d3_1.png', d3_1)
cv2.waitKey(0)
"d3_1 噪声水平2"
img_Blur_2_7 = cv2.blur(salt_noise_img2, (7, 7))  # 均值滤波
img_medianBlur_2_7 = cv2.medianBlur(salt_noise_img2, 7)  # 中值滤波
d3_2 = np.hstack((salt_noise_img2, img_Blur_2_7, img_medianBlur_2_7))
cv2.imshow('d3_2', d3_2)
cv2.imwrite('d3_2.png', d3_2)
cv2.waitKey(0)
"d3_3 噪声水平3"
img_Blur_3_7 = cv2.blur(salt_noise_img3, (7, 7))  # 均值滤波
img_medianBlur_3_7 = cv2.medianBlur(salt_noise_img3, 7)  # 中值滤波
d3_3 = np.hstack((salt_noise_img3, img_Blur_3_7, img_medianBlur_3_7))
cv2.imshow('d3_3', d3_3)
cv2.imwrite('d3_3.png', d3_3)
cv2.waitKey(0)







