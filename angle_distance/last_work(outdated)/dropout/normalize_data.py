import math
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext, Row
from pyspark.sql.types import *

conf = SparkConf().setAppName("normalize_data")
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)


def normalize_tanh(data, mean, stddev):
    data = float(data)
    return str(0.5*(math.tanh(0.01 * ((data - mean)/stddev)) + 1))

def normalize_0_1(data, max, min):
    data = float(data)
    return str( (data-min)/(max-min) )

def normalize_max_division(data, max):
    return str (float(data)/max)


def normalize():
    file_train = sc.textFile(
        "/home/ariel/Dropbox/UDC/angle_distance/dataset/leftArmMovement.txt")
    file_train = file_train.map(lambda line: line.split(" "))
    file_train = file_train.map(lambda line: [normalize_max_division(line[0], 2.69), 
                                              normalize_max_division(line[1], math.pi),  
                                              normalize_max_division(line[2], math.pi), 
                                              line[3],
                                              line[4]])
    new_file_train = file_train.map(lambda line: ' '.join(line))
    new_file_train.repartition(1).saveAsTextFile(
        "./X.txt")

    file_test = sc.textFile(
        "/home/ariel/Dropbox/UDC/angle_distance/dataset/data_last_500.txt")
    file_test = file_test.map(lambda line: line.split(" "))
    file_test = file_test.map(lambda line: [normalize_max_division(line[0], 2.69), 
                                              normalize_max_division(line[1], math.pi),  
                                              normalize_max_division(line[2], math.pi),
                                              line[3],
                                              line[4]])
    new_file_test = file_test.map(lambda line: ' '.join(line))
    new_file_test.repartition(1).saveAsTextFile(
        "./Y.txt")
    '''
    file_train = sc.textFile(
        "/home/ariel/Dropbox/UDC/angle_distance/dataset/data_last_500.txt")
    file_train = file_train.map(lambda line: line.split(" "))
    file_train = file_train.map(lambda line: [normalize_tanh(line[0], 0.9157062170992848, 0.49614328875937913), 
                                              normalize_tanh(line[1], -0.0008359381960645428, 1.7221525641385096),  
                                              normalize_tanh(line[2], 0.0006710009394591427, 1.8143846105666046), 
                                              line[3],
                                              line[4]])
    new_file_train = file_train.map(lambda line: ' '.join(line))
    new_file_train.repartition(1).saveAsTextFile(
        "./X.txt")
        '''

def add_columns():
    file = sc.textFile("./leftArmMovement.txt").map(lambda line: line.split(" ")
                                                    ).map(lambda line: [float(i) for i in line])
    schema = StructType([
        StructField("d", FloatType(), False),
        StructField("a", FloatType(), False),
        StructField("ac", FloatType(), False),
        StructField("d1", FloatType(), False),
        StructField("a1", FloatType(), False)
    ])
    file = sqlContext.createDataFrame(file, schema)
    file.describe(["d"]).show()
    #file.select(mean("a")).collect()

normalize()
