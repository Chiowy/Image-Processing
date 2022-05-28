import numpy as np
from matplotlib import pyplot as plt


def equalizehist(lenofhist: int, hist: np.ndarray, shape: [int, int]):
    """
    equalize the hist
    :param lenofhist: 输入图像的灰度级，直方图的size
    :param hist: 要 equalized的直方图
    :param shape: 原属图像的长宽
    :return: equalized hist
    """
    hist_after_equalized = np.zeros(len(hist))  # 建立一个空的直方图，维度为size
    length, width = shape  # 获取尺寸
    r_k = hist / (length * width)  # 概率分布
    s_k = np.zeros(len(hist))  # 累计概率分布
    for i in range(lenofhist):
        if i == 0:
            s_k[i] = r_k[i]
        else:
            s_k[i] = r_k[i] + s_k[i - 1]
    for i in range(lenofhist):
        s_k[i] = int((lenofhist - 1) * s_k[i] + 0.5)  # 取整扩展
        hist_after_equalized[int(s_k[i])] += r_k[i]  # 进行映射

    return hist_after_equalized


if __name__ == '__main__':
    hist = np.array(
        [5, 12, 35, 31, 8, 6, 3]
    )
    shape = [10, 10]
    hist_after_equalized = equalizehist(len(hist), hist, shape)
    plt.plot(hist_after_equalized)
    print(hist_after_equalized)
    plt.show()
