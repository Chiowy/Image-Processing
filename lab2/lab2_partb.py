"""
使用Prewitt 算子的图像分割实验
使用Prewitt 算子进行 part a 中的全部步骤。
"""
import cv2
import numpy as np

img_gray = cv2.imread('JWEI.png', 0)

Prewitt_kernel_X = np.array(
    [[1, 1, 1],
     [0, 0, 0],
     [-1, -1, -1]]
)
Prewitt_kernel_Y = np.array(
    [[-1, 0, 1],
     [-1, 0, 1],
     [-1, 0, 1]]
)

Gx = cv2.filter2D(img_gray, cv2.CV_16S, Prewitt_kernel_X)
Gy = cv2.filter2D(img_gray, cv2.CV_16S, Prewitt_kernel_Y)
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