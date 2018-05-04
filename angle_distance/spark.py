from pyspark import SparkContext, SparkConf
from decimal import Decimal
import math
from pyspark.sql import SparkSession

PRESITION = 10

conf = SparkConf().setAppName('HandleData')
sc = SparkContext(conf=conf)

def summary(line):
    data = line[1]
    data = data.split(",")
    data = [Decimal(float(i)) for i in data]
    size = len(data)
    mean = sum(data)/size
    
    var = sum( [ Decimal(i**2) for i in data ] ) / size

    var = round(var, PRESITION)
    
    var = var - round(mean ** 2, PRESITION)
    stdv = math.sqrt(var)

    increment_percentage = 0.0
    for num in range(size-1):
        dif = round(Decimal(data[num]), PRESITION)-round(Decimal(data[num+1]), PRESITION)
        increment_percentage = abs( (dif*100)/round(Decimal(data[num]), PRESITION) )
    increment_percentage /= size
    return line[0]+","+str(round(mean, PRESITION))+","+str(var)+","+str(stdv)+","+str(increment_percentage)


file = sc.textFile("./weights_logs.csv")
file = file.map(lambda line: line.split(","))
file = file.map(lambda line: ( line[1]+"_"+line[2], float(line[3] ) ) )
file = file.reduceByKey( lambda x, y: str(x)+","+str(y) )
file = file.sortByKey()
file.filter

summary_variables_header = "variable,mean,var,stddv,change_%"
summary_variables = file.map(summary)

weights = file.map(lambda line: ",".join(line))
weights.repartition(1).saveAsTextFile("./weights")

summaries = summary_variables.map( lambda line: "".join(line))
summaries.repartition(1).saveAsTextFile("./summaries")