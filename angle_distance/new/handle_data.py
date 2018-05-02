from pyspark import SparkContext, SparkConf
import math

conf = SparkConf().setAppName("handling_data")
sc = SparkContext(conf=conf)


def create_training_testing_files(path_file, new_path, training_percentage):
    f = sc.textFile(path_file)
    header = f.first()
    f = f.filter(lambda line: line!=header)
    train, test = f.randomSplit([training_percentage, 1-training_percentage])
    train = sc.parallelize([header]).union(train)
    test = sc.parallelize([header]).union(test)
    train.repartition(1).saveAsTextFile(new_path+"_train.csv")
    test.repartition(1).saveAsTextFile(new_path+"_test.csv")


def polar_2_cart_aux(line):
    line = [float(i) for i in line.split(",")]

    x1 = line[0]*math.cos(line[1])
    y1 = line[0]*math.sin(line[1])

    xact = 0.05*math.cos(line[2])
    yact = 0.05*math.sin(line[2])

    xfinal = line[3]*math.cos(line[4])
    yfinal = line[3]*math.sin(line[4])

    string = [x1, y1, xact, yact, xfinal, yfinal]
    string = [str(i) for i in string]
    string = ",".join(string)
    
    return string 

def polar_2_cart(path_file, new_path):
    f = sc.textFile(path_file)
    header = f.first()
    f = f.filter(lambda line: line!=header)
    f = f.map(polar_2_cart_aux)
    f.repartition(1).saveAsTextFile(new_path)

create_training_testing_files("../dataset/cart_leftArmMovement.csv", "./cart_leftArmMovement", 0.001)