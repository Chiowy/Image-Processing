import numpy as np
from code.dig import myfunc

# 用sobel算子和拉普拉斯算子处理以下矩阵
matrix = np.array(
    [[5, 5, 1, 6, 4],
    [7, 5, 2, 7, 7],
    [5, 8, 16, 7, 5],
    [6, 16, 8, 23, 6],
    [9, 8, 1, 4, 5]]
)

sobel_x = np.array(
    [[-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]]
)
sobel_y = np.array(
    [[1,  2,  1],
    [0,  0,  0],
    [-1, -2, -1]]
)

laplace_kernel = np.array(
    [[0, 1, 0],
     [1, -4, 1],
     [0, 1, 0]]
)

myfunc = myfunc.Myfunc()

Gx = myfunc.convole_handy(matrix, sobel_x)
Gy = myfunc.convole_handy(matrix, sobel_y)
G = Gx + Gy

L = myfunc.convole_handy(matrix, laplace_kernel)
print(G)
print(L)
