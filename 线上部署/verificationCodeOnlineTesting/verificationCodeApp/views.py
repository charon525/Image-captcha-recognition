from django.shortcuts import render
import cv2
from cv2 import HOGDescriptor
import numpy as np
import os
import urllib.request
from django.http import JsonResponse
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark import SparkContext, SparkConf
from elephas.spark_model import SparkModel, load_spark_model  

characters = ['1', '2', '3', '4', '5', '6', '7', '8', '9',
              'a', 'b', 'd', 'e', 'f', 'g', 'h', 'i', 'j',  'l', 'm', 'n', 'q', 'r', 't',  'y',
              'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
char_index = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6, '8': 7, '9': 8,
              'a': 9, 'b': 10, 'd': 11, 'e': 12, 'f': 13, 'g': 14, 'h': 15, 'i': 16, 'j': 17, 'l': 18, 'm': 19, 'n': 20, 'q': 21, 'r': 22, 't': 23, 'y': 24,
              'A': 25, 'B': 26, 'C': 27, 'D': 28, 'E': 29, 'F': 30, 'G': 31, 'H': 32, 'I': 33, 'J': 34, 'K': 35, 'L': 36, 'M': 37,'N': 38, 'O': 39, 'P': 40, 'Q': 41, 'R': 42, 'S': 43, 'T': 44, 'U': 45, 'V': 46, 'W': 47, 'X': 48, 'Y': 49, 'Z': 50}
index_char = {0: '1', 1: '2', 2: '3', 3: '4', 4: '5', 5: '6', 6: '7', 7: '8', 8: '9',
            9: 'a', 10: 'b', 11: 'd', 12: 'e', 13: 'f', 14: 'g', 15: 'h', 16: 'i', 17: 'j', 18: 'l', 19: 'm', 20: 'n', 21: 'q', 22: 'r', 23: 't', 24: 'y',
            25: 'A', 26: 'B', 27: 'C', 28: 'D', 29: 'E', 30: 'F', 31: 'G', 32: 'H', 33: 'I', 34: 'J', 35: 'K', 36: 'L', 37: 'M', 38: 'N', 39: 'O', 40: 'P', 41: 'Q', 42: 'R', 43: 'S', 44: 'T', 45: 'U', 46: 'V', 47: 'W', 48: 'X', 49: 'Y', 50: 'Z'}
CNN_Model =  load_spark_model(r'./models/Kerasmodel_CNN96.h5')

winSize = (32, 64)  # 定义检测窗口的大小，通常表示为像素大小。
blockSize = (8, 16) # 指定了每个块（block）的像素大小，用于对图像进行局部归一化。
blockStride = (8, 16) # 指定了块的步进（stride），即两个相邻块之间的距离。
cellSize = (4, 8) # 定义了每个单元（cell）的像素大小。
nbins = 9 # 指定了直方图中的箱子（bin）数量，用于表示梯度的方向。
derivAperture = 1 # 表示用于计算图像梯度的Sobel核的孔径大小。
winSigma =  4 # 表示高斯平滑窗口的标准差。
historgramNormType = 0 # 指定直方图的归一化类型，通常为0或1。
L2HysThreshold = 2.0000000000000001e-01  # 指定用于阈值化梯度幅值的阈值
gammaCorrection = 0 # 指定是否进行Gamma校正。
nlevels = 64 # 表示HOG特征金字塔的层数。
hog = HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture,
                    winSigma,  historgramNormType, L2HysThreshold, gammaCorrection, nlevels)


def RemoveSmallCC(img, small, connectivity=8):
    ret, labels = cv2.connectedComponents(img, connectivity=connectivity)
    for n in range(ret + 1):  # 0 ~ max
        num = 0  # 清零
        for elem in labels.flat:
            if elem == n:
                num += 1
        if num < small:  # 去除小连通域
            img[np.where(labels == n)] = 0
    return img


def projection(img):
    img_tmp = img / 255.0
    proj = np.zeros(shape=(img_tmp.shape[1]))
    for col in range(img_tmp.shape[1]):
        v = img_tmp[:, col]
        proj[col] = np.sum(v)
    return proj


def Removeline(img):
    proj = projection(img)
    flaw_area = np.where(proj < 5)
    img[:, flaw_area] = 0  # 去除横线
    return RemoveSmallCC(img, 200)


def Extract_num(proj):
    num = 0
    COUNT, LOC = [], []
    for i in range(len(proj)):
        if proj[i]:  # 如果非零则累加
            num += 1
            if i == 0 or proj[i-1] == 0:  # 记录片段起始位置
                start = i
            if i == len(proj) -1 or proj[i+1] == 0:  # 定义派那段结束标志，并记录片段
                end = i
                if num > 10:   # 提取有效片段
                    COUNT.append(num)
                    LOC.append((start, end))
                num = 0
    return LOC, COUNT


def Segment4_Num(COUNT, LOC):
    if len(COUNT) > 4 or len(COUNT) <= 0:
        return
    if len(COUNT) == 4:  # (1,1,1,1)
        return LOC
    if len(COUNT) == 3:  # (1,1,2)
        idx = np.argmax(np.array(COUNT))  # 最大片段下标
        r = LOC[idx]  # 最大片段位置
        start = r[0]
        end = r[1]
        m = (r[0] + r[1]) // 2  # 中间位置
        # 修改LOC[idx]
        LOC[idx] = (start, m)
        LOC.insert(idx + 1, (m + 1, end))
        return LOC
    if len(COUNT) == 2:  # (2,2)or(1,3)
        skew = max(COUNT) / min(COUNT)  # 计算偏移程度
        if skew < 1.7:  # 认为是（2，2）
            start1 = LOC[0][0]
            end1 = LOC[0][1]
            start2 = LOC[1][0]
            end2 = LOC[1][1]
            m1 = (start1 + end1) // 2
            m2 = (start2 + end2) // 2
            return [(start1, m1), (m1 + 1, end1), (start2, m2), (m2 + 1, end2)]
        else:  # 认为是（1，3）
            idx = np.argmax(np.array(COUNT))  # 最大片段下标
            start = LOC[idx][0]
            end = LOC[idx][1]
            m1 = (end - start) // 3 + start
            m2 = (end - start) // 3 * 2 + start
            LOC[idx] = (start, m1)
            LOC.insert(idx + 1, (m1 + 1, m2))
            LOC.insert(idx + 2, (m2 + 1, end))
            return LOC
    if len(COUNT) == 1:  # (4)
        start = LOC[0][0]
        end = LOC[0][1]
        m1 = (end - start) // 4 + start
        m2 = (end - start) // 4 * 2 + start
        m3 = (end - start) // 4 * 3 + start
        return [(start, m1), (m1 + 1, m2), (m2 + 1, m3), (m3 + 1, end)]


def ImageProcessing( img):
        #cv_show( img, 'origin')
        img = cv2.resize(img, (160, 60))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 首先对图像进行灰度处理
        _, img = cv2.threshold(img, int(0.9 * 255), 255, cv2.THRESH_BINARY_INV)  # 二值化操作
        img = RemoveSmallCC(img, 200, connectivity=4)  # 去除小连通域
        kernel_erode = np.ones((3, 3), np.uint8)  # 进行图片腐蚀操作
        img = cv2.erode(img, kernel_erode, iterations=1)
        img = RemoveSmallCC(img, 30)  # 去除小连通域
        kernel_dilate = np.ones((3, 3), np.uint8)  # 进行图片膨胀操作
        img = cv2.dilate(img, kernel_dilate, iterations=1)
        img = Removeline(img)
        proj = projection(img)
        loc, count = Extract_num(proj)
        LOC = Segment4_Num(count, loc)
        NUM0 = RemoveSmallCC(img[:, LOC[0][0]:LOC[0][1]], 50)
        NUM1 = RemoveSmallCC(img[:, LOC[1][0]:LOC[1][1]], 50)
        NUM2 = RemoveSmallCC(img[:, LOC[2][0]:LOC[2][1]], 50)
        NUM3 = RemoveSmallCC(img[:, LOC[3][0]:LOC[3][1]], 50)
        img0, img1, img2, img3 = cv2.resize(NUM0, (25,60)), cv2.resize(NUM1, (25,60)), cv2.resize(NUM2, (25,60)), cv2.resize(NUM3, (25,60))
        return [img0, img1, img2, img3]


def img_hog(images):
    features = []
    for img in images:
        img_tmp = cv2.resize(img, (32, 64))
        img_arr = hog.compute(img_tmp)
        features.append(np.array(img_arr).reshape(576))
    return features

def CNN_predict(images):
    imgwitherode = []
    kernel_erode = np.ones((3, 3), np.uint8)  # 进行图片腐蚀操作
    for img in images:
        img_erode = cv2.erode(img, kernel_erode, iterations=1)
        imgwitherode.append(img_erode)
    predictions = CNN_Model.predict(np.array(imgwitherode))
    for i in range(len(predictions)):
        predic_txt = ''
        predic_txt += index_char[list(predictions[0]).index(max(list(predictions[0])))]
        predic_txt += index_char[list(predictions[1]).index(max(list(predictions[1])))]
        predic_txt += index_char[list(predictions[2]).index(max(list(predictions[2])))]
        predic_txt += index_char[list(predictions[3]).index(max(list(predictions[3])))]
    return predic_txt

tmp = cv2.imread('./1a1V.jpg')
CNN_predict(ImageProcessing(tmp))

def index(request):
    return render(request, 'index.html')


def process_image(request):
    if request.is_ajax():
        image_url = request.GET.get('img')  # 假设前端通过POST请求将图片地址传回后端
        req = urllib.request.urlopen(image_url)
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        img = cv2.imdecode(arr, -1)  # 使用OpenCV读取图片
        subimgs = ImageProcessing(img)
        res = CNN_predict(subimgs)
        response = JsonResponse({"res":res})
        return response
