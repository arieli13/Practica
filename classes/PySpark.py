from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, DataFrameReader
import math
from pyspark.sql.functions import stddev, mean, col



class PySpark:

    def __init__(self, app_name):
        conf = SparkConf()
        self.__spark = SparkSession.builder \
             .master("local") \
             .appName(app_name) \
             .config(conf=conf) \
             .getOrCreate()
        

    def read_csv(self, path, separator, header):
        return self.__spark.read.csv(path, sep=separator, header=header, inferSchema=True)
    
    
    def split_file(self, split_percentage, rdd):
        x, y = rdd.randomSplit([split_percentage, 1-split_percentage])
        return x, y

    def normalize(self, df, columns):
        selectExpr = []
        cont = 0
        for column in columns:
            average = df.agg(mean(df[column]).alias("mean")).collect()[0]["mean"]
            stddev_ = df.agg(stddev(df[column]).alias("stddev")).collect()[0]["stddev"]
            if cont > 3:
                selectExpr.append(df[column])
            else:
                selectExpr.append((df[column] - average)/(stddev_))
            cont += 1
        return df.select(selectExpr)

    def main(self):
        f = self.read_csv("../datasets_angle_distance/leftArmMovement.csv", ",", True)
        x = self.normalize(f, ["distance_t", "angle_t", "angle_act", "distance_t+1", "angle_t+1"])
        rdd = x.rdd
        rdd = rdd.map(lambda line: ",".join([str(i) for i in line]))
        rdd.repartition(1).saveAsTextFile("abc")

x = PySpark("y")
x.main()