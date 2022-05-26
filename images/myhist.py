
import numpy
import numpy as np

def img2hist(img_gray : numpy.ndarray): # 输入图像矩阵（256），输出直方图
    hist = np.arange(256) # imread 读取的默认范围是 0 - 255，所以直方图的维度为256
    for i in img_gray:
        for j in i:
            hist[img_gray[i][j]] += 1

    return hist