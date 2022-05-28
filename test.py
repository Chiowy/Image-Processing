import cv2
import numpy as np


x = np.array([[1.4, 1.4, 1.4, 1.3],
             [1.4, 1.4, 1.4, 1.3],
             [1.4, 1.4, 1.4, 1.3]])
y = np.array([[2.2, 1.2, 1.3, 1.3],
             [1.4, 1.4, 1.4, 1.3],
             [1.4, 1.4, 1.4, 1.3]])
print(x)
print(y)

z = np.asarray([x, y])
print(z)