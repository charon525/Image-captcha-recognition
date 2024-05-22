#!/usr/bin/env  python3
from pyspark import SparkContext, SparkConf
from keras.models import Sequential
from keras.optimizers import Adam
import numpy as np
import os
import time
import cv2
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.convolutional import MaxPooling2D
from elephas.spark_model import SparkModel
from keras.layers.core import Dense, Flatten, Dropout
from keras.callbacks import TensorBoard
from elephas.utils.rdd_utils import to_simple_rdd

nums = []
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
conf = SparkConf().setAppName('Elephas_CNN').setMaster('local')
sc = SparkContext(conf=conf)
sc.setLogLevel("WARN")  # 设置日志级别
sql_context = SQLContext(sc)
spark = SparkSession.builder.appName("KerasModel").getOrCreate()
path = "/home/hadoop/proj"


def readfile(path):
    dirs = sorted(os.listdir(path))
    x = []
    y = []
    for index, dir in enumerate(dirs):
        path1 = os.path.join(path, dir)
        images = os.listdir(path1)  # 将path路径下的文件名以列表形式读出
        for num, img in enumerate(images):
            path2 = os.path.join(path1, img)
            image = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
            # 进行图片腐蚀操作
            image = cv2.erode(image, kernel_erode, iterations=1)
            image = cv2.resize(image, (25, 60))
            x.append(image)
            y.append(np.eye(51)[np.array([char_index[dir[0]]])])
    # x_normalized = [np.array(image)/255.0 for image in x]
    return np.array(x), np.squeeze(np.array(y).astype(np.uint8), axis=1)


def predict(spark_model, x_test, y_test):
    predictions = spark_model.predict(x_test)  # perform inference
    # evaluation = spark_model.evaluate(x_test, y_test) # perform evaluation/scoring
    TrueNum = 0
    for i in range(len(y_test)):
        index1 = list(y_test[i]).index(max(list(y_test[i])))
        index2 = list(predictions[i]).index(max(list(predictions[i])))
        if index1 != index2:
            print(
                "预测：{}， 实际：{}， 概率为：{}".format(index_char[index2], index_char[index1], max(list(predictions[i]))))
        else:
            TrueNum += 1
    print("正确数量：{}， 总数{}， acc： {} ".format(TrueNum, len(y_test), TrueNum / len(y_test)))
    return TrueNum / len(y_test)


def init_model(nb_classes):
    model = Sequential()
    model.add(ZeroPadding2D(padding=((1, 2), (2, 1)), input_shape=(60, 25, 1)))
    model.add(Conv2D(32, kernel_size=(3, 3), padding="valid", activation="relu"))
    model.add(BatchNormalization())
    # model.add((Activation("relu")))
    model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))

    model.add(Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu"))
    # model.add((Activation("relu")))
    model.add(BatchNormalization())
    # model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))

    model.add(Conv2D(64, kernel_size=(5, 5), padding="same", activation="relu"))
    # model.add((Activation("relu")))
    model.add(BatchNormalization())
    # model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))

    model.add(Conv2D(32, kernel_size=(5, 5), padding="same", activation="relu"))
    model.add(BatchNormalization())
    # model.add((Activation("relu")))
    model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))

    model.add(Flatten())
    model.add(Dense(1024, activation="relu"))

    model.add(Dense(nb_classes, activation='softmax'))

    return model


kernel_erode = np.ones((3, 3), np.uint8)  # 腐蚀操作的核

if __name__ == "__main__":
    x_test, y_test = readfile(path + '/test')
    x_train, y_train = readfile(path + '/train')

    print("----------------------->Load data successfully!")
    rdd = to_simple_rdd(sc, x_train, y_train)
    print("----------------------->Get rdd successfully!")
    model = init_model(51)
    opt = Adam(learning_rate=1e-3)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy'])
    spark_model = SparkModel(model, frequency='epoch', mode='asynchronous')
    print("----------------------->Train start!")
    tensorboard = TensorBoard(log_dir='./logs')
    start_time = time.time()
    spark_model.fit(rdd, epochs=4, batch_size=256, verbose=1, validation_split=0, callbacks=[tensorboard])
    print("----------------------->Train sunccessfully!, Duration: {}s".format(time.time() - start_time))

    acc = predict(spark_model, x_test, y_test)
    spark_model.save('/home/hadoop/proj/Kerasmodel_CNN{}.h5'.format(int(acc * 100)))
    print("------------->Save model successfully!")
    # spark_model= load_spark_model('./Kerasmodel.h5')
    sc.stop()
    pass 