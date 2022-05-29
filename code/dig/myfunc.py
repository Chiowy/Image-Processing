import numpy as np


class Myfunc:
    def convole_handy(self,
                      matrix: np.ndarray,
                      kernel: np.ndarray):
        """
        convole 进行卷积
        :param matrix: 输入矩阵, 可能不是方阵
        :param kernel: 卷积核 3*3 or 5*5 ...， 必须是方阵
        :return:
        """
        result = matrix.copy()
        side = (len(kernel) - 1) // 2  # 整数
        for i in range(side, matrix.shape[0] - 1 - side):
            for j in range(side, matrix.shape[1] - 1 - side):
                for m in range(len(kernel)):
                    for n in range(len(kernel)):
                        result[i][j] += kernel[m][n] * matrix[i - side + m][j - side + n]

        return result

    def equalizehist(self,
                     lenofhist: int,
                     hist: np.ndarray,
                     shape: [int, int]):
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

    def img2hist(self,
                 img_gray: np.ndarray):
        """
        基本描述 img -> hist
        详细描述
           Args:
               img_gray(ndarray): 输入灰度图像

           Returns:
               hist(ndarray): 输出直方图
           """
        hist = np.arange(256)  # imread 读取的默认范围是 0 - 255，所以直方图的维度为256
        for i in img_gray:
            for j in i:
                hist[img_gray[i][j]] += 1

        return hist

