import cv2
from cv2 import HOGDescriptor
import numpy as np
import os
# from skimage.feature import hog
from sklearn.model_selection import train_test_split

winSize = (32, 64)  # 定义检测窗口的大小，通常表示为像素大小。
blockSize = (8, 16)  # 指定了每个块（block）的像素大小，用于对图像进行局部归一化。
blockStride = (8, 16)  # 指定了块的步进（stride），即两个相邻块之间的距离。
cellSize = (4, 8)  # 定义了每个单元（cell）的像素大小。
nbins = 9  # 指定了直方图中的箱子（bin）数量，用于表示梯度的方向。
derivAperture = 1  # 表示用于计算图像梯度的Sobel核的孔径大小。
winSigma = 4  # 表示高斯平滑窗口的标准差。
historgramNormType = 0  # 指定直方图的归一化类型，通常为0或1。
L2HysThreshold = 2.0000000000000001e-01  # 指定用于阈值化梯度幅值的阈值
gammaCorrection = 0  # 指定是否进行Gamma校正。
nlevels = 64  # 表示HOG特征金字塔的层数。
hog = HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture,
                    winSigma, historgramNormType, L2HysThreshold, gammaCorrection, nlevels)

characters = ['1', '2', '3', '4', '5', '6', '7', '8', '9',
              'a', 'b', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'l', 'm', 'n', 'q', 'r', 't', 'y',
              'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
              'V', 'W', 'X', 'Y', 'Z']
char_index = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6, '8': 7, '9': 8,
              'a': 9, 'b': 10, 'd': 11, 'e': 12, 'f': 13, 'g': 14, 'h': 15, 'i': 16, 'j': 17, 'l': 18, 'm': 19, 'n': 20,
              'q': 21, 'r': 22, 't': 23, 'y': 24,
              'A': 25, 'B': 26, 'C': 27, 'D': 28, 'E': 29, 'F': 30, 'G': 31, 'H': 32, 'I': 33, 'J': 34, 'K': 35,
              'L': 36, 'M': 37, 'N': 38, 'O': 39, 'P': 40, 'Q': 41, 'R': 42, 'S': 43, 'T': 44, 'U': 45, 'V': 46,
              'W': 47, 'X': 48, 'Y': 49, 'Z': 50}
index_char = {0: '1', 1: '2', 2: '3', 3: '4', 4: '5', 5: '6', 6: '7', 7: '8', 8: '9',
              9: 'a', 10: 'b', 11: 'd', 12: 'e', 13: 'f', 14: 'g', 15: 'h', 16: 'i', 17: 'j', 18: 'l', 19: 'm', 20: 'n',
              21: 'q', 22: 'r', 23: 't', 24: 'y',
              25: 'A', 26: 'B', 27: 'C', 28: 'D', 29: 'E', 30: 'F', 31: 'G', 32: 'H', 33: 'I', 34: 'J', 35: 'K',
              36: 'L', 37: 'M', 38: 'N', 39: 'O', 40: 'P', 41: 'Q', 42: 'R', 43: 'S', 44: 'T', 45: 'U', 46: 'V',
              47: 'W', 48: 'X', 49: 'Y', 50: 'Z'}
#            数字             英文字母（大小写）

kernel_erode = np.ones((3, 3), np.uint8)


def get_imageData(path0):
    dirs = sorted(os.listdir(path0))
    imgs, labels = [], []
    for dir in dirs:
        path1 = os.path.join(path0, dir)
        pic_names = sorted(os.listdir(path1))
        for pic_name in pic_names:
            imgarr = cv2.imread(os.path.join(path1, pic_name))  # opencv读图片
            imgarr = cv2.resize(imgarr, (32, 64))
            img_arr = hog.compute(imgarr)
            # hog特征计算
            label = char_index[dir[0]]  # 图片的标签
            imgs.append(img_arr)
            labels.append(label)
    return imgs, labels


def load_data(path):
    """获取特征数据集，x是feature，y是label"""
    x, y = get_imageData(path)
    x = [i.reshape(576) for i in x]
    x = np.array(x).astype(np.float)
    y = np.array(y).astype(np.float)
    return x, y


if __name__ == "__main__":

    x_train, y_train = load_data("./train")
    x_test, y_test = load_data("./test")
    print("load data successful")

    train_df = np.concatenate((x_train, np.expand_dims(y_train, 1)), 1)
    test_df = np.concatenate((x_test, np.expand_dims(y_test, 1)), 1)

    # 写成csv文件
    if not os.path.exists('src'):
        os.mkdir('src')
    np.savetxt("src/train.csv", train_df, delimiter=",", fmt="%.1f")
    np.savetxt("src/test.csv", test_df, delimiter=",", fmt="%.1f")

    print("data saved successfully!")
