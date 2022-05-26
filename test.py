import cv2
import numpy as np
from images.myhist import img2hist
from myEqualizeHist import equalizehist
from matplotlib import pyplot as plt

# 读入图片，cv2.IMREAD_UNCHANGED包含alpha通道
path = r'images/Anastasia.jpg'
img = cv2.imread(path) # 默认彩色图

img_gray = cv2.imread(path, 0) # 灰度图
# or img_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

# hist = img2hist(img_gray)  # 转换成直方图
hist_test = np.array(
    [5, 12, 35, 31, 8, 6, 3]
)
hist_after_equalized = equalizehist(7, hist_test, [10, 10])
plt.plot(hist_after_equalized)
plt.show()

"""cv2.imshow('beauty', img_gray)
cv2.waitKey(0)
"""