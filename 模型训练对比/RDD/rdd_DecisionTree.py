from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
import time

conf = SparkConf().setAppName("DecisionTree").setMaster("spark://master:7077")
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

# 设置参数
numClasses = 51  # 类别数
categoricalFeaturesInfo = {}  # 如果没有分类特征，可以为空字典
impurity = "gini"  # 不纯度计算方式
maxDepth = 5  # 决策树最大深度
maxBins = 32  # 分割时考虑的最大箱数
minInstancesPerNode = 1  # 父分割所需的子节点最小实例数
minInfoGain = 0.0  # 创建分割所需的最小信息增益
# 训练模型
print("model train start at:", time.strftime('%Y-%m-%d %H:%M:%S'))
model = DecisionTree.trainClassifier(rdd_train, numClasses=numClasses,
                                     categoricalFeaturesInfo=categoricalFeaturesInfo,
                                     impurity=impurity, maxDepth=maxDepth,
                                     maxBins=maxBins, minInstancesPerNode=minInstancesPerNode,
                                     minInfoGain=minInfoGain)
print("model train successful at:", time.strftime('%Y-%m-%d %H:%M:%S'))





## 计算准确率
predictions =model.predict(rdd_test.map(lambda x:x.features))
labels_and_predictions = rdd_test.map(lambda x:x.label).zip(predictions)
acc = labels_and_predictions.filter(lambda x:x[0] == x[1]).count() / float(rdd_test.count())
print("Model accuracy: %.3f%%"%(acc*100))

## 保存模型
path = '/proj/DecisionTree' + '_' + str(acc*100)
model.save(sc, path)
print("Model saved at: ",path)
sc.stop()


