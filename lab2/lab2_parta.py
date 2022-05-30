import numpy as np
import cv2

""" part a
调入并显示图像；使用Sobel算子对图像进行边缘检测处理； 
由于sobel算法由两个模板，所以分别显示处理后的水平边界和垂直边界检测结果；
用“欧几里德距离”和“街区距离”方式计算梯度的模，并显示检测结果；
对于检测结果进行二值化处理，并显示处理结果；
"""
img_gray = cv2.imread("E:\\CODE\\Image-Processing\\lab2\\JWEI.png", 0)
cv2.imshow('img', img_gray)
cv2.waitKey(0)  # 调入并显示图像

Gx = cv2.Sobel(img_gray, cv2.CV_16S, 1, 0, ksize=3)  # 当图像深度是 np.uint8 的时候，负值就会变成 0，基于这个原因，需要把输出图像的数据类型设置高一些
Gy = cv2.Sobel(img_gray, cv2.CV_16S, 0, 1, ksize=3)
Gx = cv2.convertScaleAbs(Gx)  # 处理完毕之后，在通过 cv2.convertScaleAbs函数将其转回原来的 uint8 格式
Gy = cv2.convertScaleAbs(Gy)

stackOfGxAndGy = np.hstack((Gx, Gy))
cv2.imshow('result of x and y', stackOfGxAndGy)
cv2.waitKey(0)  # 显示处理后的水平边界和垂直边界检测结果

City_Block_Distance = Gx + Gy  # 计算城区距离
# 先进行归一化，0-255 -> 0-1, 得到float64类型的数据
Gx_Normalized = Gx / 255
Gy_Normalized = Gy / 255
Euclidean_distance = np.sqrt(np.power(Gx_Normalized, 2) + np.power(Gy_Normalized, 2))  # 计算欧式距离
Euclidean_distance = (Euclidean_distance * 255).astype(np.uint8)  # 转换回去

stack_EuclideanAndCity_Block_Distance = np.hstack((Euclidean_distance, City_Block_Distance))
cv2.imshow('result of Euclidean and City_Block', stack_EuclideanAndCity_Block_Distance)
cv2.waitKey(0)  # 显示检测结果

# 使用OpenCV函数找到otsu的阈值
ret1, stackOfGxAndGy_otsu = cv2.threshold(stackOfGxAndGy, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
ret2, stack_EuclideanAndCity_Block_Distance_ostu = cv2.threshold(stack_EuclideanAndCity_Block_Distance, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
stack_otsu = np.hstack((stackOfGxAndGy_otsu, stack_EuclideanAndCity_Block_Distance_ostu))
cv2.imshow('otsu', stack_otsu)
cv2.waitKey(0)

