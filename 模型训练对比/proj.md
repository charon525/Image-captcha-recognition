### 说明：
1. 运行mkdirs创建文件夹
2. 运行GetImage.py生成9000张验证码图片
3. 运行ImagePreprocessing.py切割裁剪图片
4. hogFeatures进行HOG特征提取，生成csv文件
4. web项目为：verificationCodeOnlineTesting


###  启动集群
cd /usr/local/hadoop  
sbin/start-all.sh 

cd /usr/local/spark
sbin/start-master.sh -h master
sbin/start-slave.sh spark://master:7077
sbin/start-slaves.sh spark://master:7077

### 提交代码
bin/spark-submit --master spark://master:7077 --executor-memory 2048M /home/hadoop/proj/code.py


### 上传文件
bin/hadoop fs -put /home/hadoop/proj/src/train.csv /proj/
bin/hadoop fs -put /home/hadoop/proj/src/test.csv /proj/

### 拉取文件到本地
cd /usr/local/hadoop/
bin/hadoop fs -get /proj/LogisticRegression_{} /home/hadoop/proj/

### 显示proj中的文件
hdfs dfs -ls -R /proj/

### 删除文件
hdfs dfs -rm -r /proj/D*

### 运行web项目
cd /home/hadoop/proj/verificationCodeOnlineTesting
python3 manage.py runserver 0.0.0.0:8080 --noreload