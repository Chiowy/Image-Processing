import cv2
import numpy
import numpy as np
from matplotlib import pyplot as plt

def img2hist(img_gray : numpy.ndarray):
    """
    基本描述 img -> hist
    详细描述
       Args:
           img_gray(ndarray): 输入灰度图像

       Returns:
           hist(ndarray): 输出直方图
       """
    hist = np.arange(256) # imread 读取的默认范围是 0 - 255，所以直方图的维度为256
    for i in img_gray:
        for j in i:
            hist[img_gray[i][j]] += 1

    return hist

if __name__ == '__main__':  # for test
    img_gray = cv2.imread('E:\CODE\Image-Processing\images\Anastasia.jpg', 0)
    hist = img2hist((img_gray))
    plt.plot(hist)
    plt.show()