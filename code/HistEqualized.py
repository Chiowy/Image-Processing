import cv2
import numpy as np
from matplotlib import pyplot as plt

beauty_gray = cv2.imread('E:\CODE\Image-Processing\images\Anastasia.jpg', 0)  # 读取图像

hist = cv2.calcHist(beauty_gray, [0], None, [256], [0, 255])  # 获得直方图
"""
参数1：要计算的原图，以方括号的传入，如：[img]。
参数2：类似前面提到的dims，灰度图写[0]就行，彩色图B/G/R分别传入[0]/[1]/[2]。
参数3：要计算的区域ROI，计算整幅图的话，写None。
参数4：也叫bins,子区段数目，如果我们统计0-255每个像素值，bins=256；如果划分区间，比如0-15, 16-31…240-255这样16个区间，bins=16。
参数5：range,要计算的像素值范围，一般为[0,256)。
"""

"""
或者可以将img扁平化，再用matplotlib自带的绘制直方图的函数绘制直方图
plt.hist(beauty_gray.ravel(), 256, [0, 255])
"""


img_after_equalized = cv2.equalizeHist(beauty_gray)  # 直方图均衡化
hist_after_equalized = cv2.calcHist(img_after_equalized, [0], None, [256], [0, 255])  # 获得均衡化的直方图

"""展示img"""
befor_and_after = np.hstack((beauty_gray, img_after_equalized))  # stack 2 imgs side by side
cv2.imshow('befor and after', befor_and_after)
cv2.waitKey(0)

"""展示直方图"""
fig, ax = plt.subplots()  # 创建图实例
ax.plot(hist, color='r', label='before')
ax.plot(hist_after_equalized, color='g', label='after')
ax.legend()  # 显示lable
plt.show()
