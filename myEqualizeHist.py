import numpy as np

def equalizehist(size:int, hist:np.ndarray, shape:[int, int]):
    hist_after_equalized = np.zeros(size) # 建立一个空的直方图，维度为size
    length, width = shape  # 获取尺寸
    r_k = hist / (length * width)  # 概率分布
    s_k = np.zeros(size) # 累计概率分布
    for i in range(size):
        if i == 0: s_k[i] = r_k[i]
        else:
            s_k[i] = r_k[i] + s_k[i - 1]
    for i in range(size):
        s_k[i] = int((size - 1) * s_k[i] + 0.5)  # 取整扩展
        hist_after_equalized[int(s_k[i])] += r_k[i]

    return hist_after_equalized




