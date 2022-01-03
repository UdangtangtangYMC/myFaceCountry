import pandas as pd
from tensorflow.keras.utils import to_categorical
import numpy as np
import cv2
import glob

img_row = 112
img_col = 112
class_name = ['한국', '일본', '중국', '미국', '아랍', '아프리카', '동남아']
img_path = 'C:\\Users\\신현호\\Downloads\\ms1m_align_112\\complete'


def load_label():
    label = pd.read_csv('C:\\Users\\신현호\\Documents\\face_label.csv')
    categorical = to_categorical(label, 7)

    return categorical


def load_img():
    glob_result = glob.glob(img_path + '/**.jpg', recursive=True)
    print(glob_result[:10])
    #imgs = [np.asarray(cv2.imread(np.fromfile(img, np.uint8))) for img in glob_result]
    #print(imgs[:10])

    #return imgs


if __name__ == '__main__':
    load_img()
