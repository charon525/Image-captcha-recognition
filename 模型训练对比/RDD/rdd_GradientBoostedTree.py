from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import Vectors
import random
from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel
import time

conf = SparkConf().setAppName("LogisticRegression").setMaster("spark://master:7077")
sc = SparkContext(conf=conf).getOrCreate()
sc.setLogLevel("WARN")   # 设置日志级别
spark = SparkSession(sc)

print("load spark successful")

TRAINPATH = "/proj/train.csv"
TESTPATH = "/proj/test.csv"

def GetParts(line):
    parts = line.split(',')
    return LabeledPoint(float(parts[-1]),Vectors.dense(parts[:-1]))

rdd_train = sc.textFile(TRAINPATH)
rdd_test = sc.textFile(TESTPATH)

rdd_train = rdd_train.map(lambda line:GetParts(line))
rdd_test = rdd_test.map(lambda line:GetParts(line))

print("load hdfs data successful")

# 数据特征维度
numFeatures = 576
# 类别数量
numClasses = 51
# 迭代次数
numIterations = 100
# 学习率
learningRate = 0.1
# 树的最大深度
maxDepth = 8  # 这里我选择了一个稍大的深度，你也可以根据需求调整
# 最大分箱数
maxBins = 64  # 这里根据特征维度选择了一个稍大的值，你也可以根据实际情况调整

## 训练逻辑回归多分类器
print("model train start at:", time.strftime('%Y-%m-%d %H:%M:%S'))
model = GradientBoostedTrees.trainClassifier(rdd_train, categoricalFeaturesInfo={}, numIterations=numIterations,
                                             learningRate=learningRate, maxDepth=maxDepth, maxBins=maxBins)
print("model train successful at:", time.strftime('%Y-%m-%d %H:%M:%S'))



## 计算准确率
predictions =model.predict(rdd_test.map(lambda x:x.features))
labels_and_predictions = rdd_test.map(lambda x:x.label).zip(predictions)
acc = labels_and_predictions.filter(lambda x:x[0] == x[1]).count() / float(rdd_test.count())
print("Model accuracy: %.3f%%"%(acc*100))

## 保存模型
path = '/proj/GradientBoostedTrees' + '_' + str(acc*100)
model.save(sc, path)
print("Model saved at: ",path)
sc.stop()


