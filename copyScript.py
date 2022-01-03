import os.path as path
import numpy as np
import cv2

img_path = 'D:\\213.jpg'

imread = np.asarray(cv2.imread(img_path))
print(imread.shape)

