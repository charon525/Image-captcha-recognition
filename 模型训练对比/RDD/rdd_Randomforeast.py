from pyspark import SparkConf, SparkContext
from pyspark.sql.session import SparkSession

# 本地配置
TRAIN_PATH = "./src/train.csv"  # 训练集csv
TEST_PATH = "./src/test.csv"  # 测试集csv
SPARK_URL = "local[*]"

# 分布式配置
# TRAIN_PATH = "hdfs://master:9000/pr/src/train.csv" # 训练集csv
# TEST_PATH = "hdfs://master:9000/pr/src/test.csv" # 测试集csv
# SPARK_URL = "spark://master:7077"

APP_NAME = "Random Forest Pr"

RANDOM_SEED = 13579
# TRAINING_DATA_RATIO = 0.7 # 训练集占比

# model0: 60 15 90 83.353%  model1: 100 15 16 84.154%

# 数据集加载

conf = SparkConf().setAppName(APP_NAME).setMaster(SPARK_URL)
conf.set("spark.executor.memory", "500m")
conf.set("spark.executor.cores", "4")
conf.set("spark.driver.memory", "3g")  # 这里是增加jvm的内存
conf.set("spark.driver.maxResultSize", "1g")
sc = SparkContext(conf=conf)
sc.setLogLevel('WARN')
spark = SparkSession(sc)
# spark = SparkSession.builder \
#    .appName(APP_NAME) \
#    .master(SPARK_URL) \
#    .getOrCreate()


df = spark.read \
    .options(header="false", inferschema="true") \
    .csv(TRAIN_PATH)

df1 = spark.read \
    .options(header="false", inferschema="true") \
    .csv(TEST_PATH)

print("Total number of train rows: %d" % df.count())
print("Total number of test rows: %d" % df1.count())

# 将DataFrame数据类型转换成RDD的DataFrame

from pyspark.mllib.linalg import Vectors

from pyspark.mllib.regression import LabeledPoint

transformed_df = df.rdd.map(lambda row: LabeledPoint(row[-1], Vectors.dense(row[0:-1])))
transformed_df1 = df1.rdd.map(lambda row: LabeledPoint(row[-1], Vectors.dense(row[0:-1])))

training_data, _ = transformed_df.randomSplit([1, 0], RANDOM_SEED)
test_data, _ = transformed_df1.randomSplit([1, 0], RANDOM_SEED)
# splits =[TRAINING_DATA_RATIO,1.0-TRAINING_DATA_RATIO]

# training_data,test_data =transformed_df.randomSplit(splits,RANDOM_SEED)


print("Number of training set rows:%d" % training_data.count())

print("Number of test set rows:%d" % test_data.count())

# 随机森林算法模型训练

from pyspark.mllib.tree import RandomForest

from time import *

start_time = time()

RF_NUM_TREES = 100  # 树的最大数量 80:41.2% 100:46.7% 110:49.2% 120:25.6%
RF_MAX_DEPTH = 15  # 树的最大深度
RF_NUM_BINS = 90  # 每个特征的最大离散化箱数 160
NUM_CLASSES = 51  # 共51类
model = RandomForest.trainClassifier(training_data, numClasses=NUM_CLASSES, categoricalFeaturesInfo={}, \
                                     numTrees=RF_NUM_TREES, featureSubsetStrategy='auto', impurity="gini", \
                                     maxDepth=RF_MAX_DEPTH, maxBins=RF_NUM_BINS, seed=RANDOM_SEED)

end_time = time()

elapsed_time = end_time - start_time

print("Time to train model: %.3f seconds" % elapsed_time)

# 模型保存
path = "./Rf_model"
num = 0
while True:
    try:
        model.save(sc, path + "_" + str(num))
        break
    except:
        num += 1
print("Model saved at: ", path + "_" + str(num))

# 预测和计算准确度

predictions = model.predict(test_data.map(lambda x: x.features))

labels_and_predictions = test_data.map(lambda x: x.label).zip(predictions)

acc = labels_and_predictions.filter(lambda x: x[0] == x[1]).count() / float(test_data.count())

print("Model accuracy: %.3f%%" % (acc * 100))

# #模型评估

# from pyspark.mllib.evaluation import BinaryClassificationMetrics


# strart_time = time()

# metrics =BinaryClassificationMetrics(labels_and_predictions)

# print("Area under Predicton/Recall(PR) curve:%.f"%(metrics.areaUnderPR*100))

# print("Area under Receiver Operating Characteristic(ROC) curve:%.3f"%(metrics.areaUnderROC*100))

# end_time = time()

# elapsed_time = end_time - start_time

# print("Time to evaluate model:%.3fseconds"%elapsed_time)