from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import Vectors
import random
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
import time

conf = SparkConf().setAppName("LogisticRegression").setMaster("spark://master:7077")
sc = SparkContext(conf=conf)
sc.setLogLevel("WARN")   # 设置日志级别
spark = SparkSession(sc)



TRAINPATH = "/proj/train.csv"
TESTPATH = "/proj/test.csv"

def GetParts(line):
    parts = line.split(',')
    return LabeledPoint(float(parts[-1]),Vectors.dense(parts[:-1]))

rdd_train = sc.textFile(TRAINPATH)
rdd_test = sc.textFile(TESTPATH)

rdd_train = rdd_train.map(lambda line:GetParts(line))
rdd_test = rdd_test.map(lambda line:GetParts(line))

print("------->Load Data Successfully!")


## 训练逻辑回归多分类器
startTime = time.time()
print("------->Model train start")
model = LogisticRegressionWithLBFGS().train(rdd_train, iterations=10, numClasses=51)
print("------->Model train successfully, Duration:{}s".format(time.time() - startTime))

## 计算准确率
scoreAndLabels = rdd_test.map(lambda point:(model.predict(point.features),point.label))
accuracy = scoreAndLabels.filter(lambda l: l[0]==l[1]).count() / rdd_test.count()
print("-------->accuracy: ",accuracy)

## 保存模型
path = '/proj/LogisticRegression' + '_' + str(accuracy*100)
model.save(sc, path)
print("-------->Model saved at hdfs, path: ",path)




