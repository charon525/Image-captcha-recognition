#!/usr/bin/env  python3
from pyspark import SparkContext, SparkConf
from keras.models import Sequential
from keras.optimizers import Adam
import numpy as np
import time
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from keras.layers import BatchNormalization
from elephas.spark_model import SparkModel, load_spark_model
from keras.layers.core import Dense, Dropout
from elephas.utils.rdd_utils import to_simple_rdd

kernel_erode = np.ones((3, 3), np.uint8)
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
# conf = SparkConf().setAppName("elephas_NN").setMaster("local")
conf = SparkConf().setAppName("elephas_NN").setMaster("spark://master:7077")
sc = SparkContext(conf=conf)
sc.setLogLevel("WARN")  # 设置日志级别
sql_context = SQLContext(sc)

# 创建Spark会话
spark = SparkSession.builder.appName("elephas_NN").getOrCreate()
# path = "/home/hadoop/proj/src/"
path = '/proj/'


def init_model(nb_classes):
    model = Sequential()
    model.add(Dense(1024, activation='relu', input_dim=576))
    model.add(BatchNormalization())
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    # model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(1024, activation="relu"))
    model.add(Dense(nb_classes, activation='softmax'))
    return model


def readfile(path):
    x = []
    y = []
    with open(path, 'r', encoding='utf-8') as lines:
        for num, line in enumerate(lines):
            parts = line.split(',')
            y.append(np.squeeze(np.eye(51)[np.array(int(float(parts[-1])))]))
            tmp = []
            for i in parts[:-1]:
                tmp.append(float(i))
            x.append(np.array(tmp))
    return np.array(x), np.array(y).astype(np.uint8)


def read_hdfs(path):
    x = []
    y = []
    text = sc.textFile(path)
    for line in text.collect():
        parts = line.split(',')
        y.append(np.squeeze(np.eye(51)[np.array(int(float(parts[-1])))]))
        tmp = []
        for i in parts[:-1]:
            tmp.append(float(i))
        x.append(np.array(tmp))
        break
    return np.array(x), np.array(y).astype(np.uint8)


def predict(spark_model, x_test, y_test):
    predictions = spark_model.predict(x_test)  # perform inference
    evaluation = spark_model.evaluate(x_test, y_test)  # perform evaluation/scoring
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


if __name__ == "__main__":
    # x_test, y_test =  readfile(path + 'test.csv')
    # x_train, y_train = readfile(path + 'train.csv')
    x_test, y_test = read_hdfs(path + 'test.csv')
    x_train, y_train = read_hdfs(path + 'train.csv')
    print("----------------------->Load data successfully!")
    rdd = to_simple_rdd(sc, x_train, y_train)
    print("----------------------->Get rdd successfully!")
    model = init_model(51)
    opt = Adam(learning_rate=1e-3)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy'])
    spark_model = SparkModel(model, frequency='epoch', mode='asynchronous')

    print("----------------------->Train start!")
    start_time = time.time()
    spark_model.fit(rdd, epochs=20, batch_size=128, verbose=1, validation_split=0)
    print("----------------------->Train sunccessfully!, Duration: {}s".format(time.time() - start_time))

    # spark_model= load_spark_model('./Kerasmodel_NN87.h5')
    acc = predict(spark_model, x_test, y_test)
    # spark_model.save('/home/hadoop/proj/Kerasmodel_NN{}.h5'.format(int(acc*100)))
    # print("Save model successfully!")

    sc.stop()
    pass 