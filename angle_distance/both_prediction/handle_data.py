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

def add_radio_aux(line):
    line = line.split(",")
    line.insert(2, "0.05")
    line = ",".join(line)
    return line

def add_radio(path_file, new_path):
    f = sc.textFile(path_file)
    header = f.first()
    f = f.filter(lambda line: line!=header)
    f = f.map(add_radio_aux)
    f.repartition(1).saveAsTextFile(new_path)

normalize_distance = lambda d: d/2.69 
normalize_angle = lambda a: (a+math.pi)/(2*math.pi)

def normalize_0_1_aux(line):
    line = [float(i) for i in line.split(",")]
    line = [ normalize_distance( line[0] ), normalize_angle( line[1] ), 
             normalize_distance( line[2] ), normalize_angle( line[3] ), 
             normalize_distance( line[4] ), normalize_angle( line[5] )   ]
    line = ",".join( [ str(i) for i in line ] )
    return line

def normalize_0_1(path_file, new_path):
    f = sc.textFile(path_file)
    header = f.first()
    f = f.filter(lambda line: line!=header)
    f = f.map(normalize_0_1_aux)
    f.repartition(1).saveAsTextFile(new_path)

def positive_angles_aux(line):
    line = [float(i) for i in line.split(",")]
    if line[1]<0:
        line[1] = line[1] + 2*math.pi
    if line[3]<0:
        line[3] = line[3] + 2*math.pi
    if line[5]<0:
        line[5] = line[5] + 2*math.pi
    line = ",".join( [ str(i) for i in line ] )
    return line

def positive_angles(path_file, new_path):
    f = sc.textFile(path_file)
    header = f.first()
    f = f.filter(lambda line: line!=header)
    f = f.map(positive_angles_aux)
    f.repartition(1).saveAsTextFile(new_path)

def add_direction_aux(line):
    line = [float(i) for i in line.split(",")]
    dir = 0
    if line[1] < 0:
        if line[5] < 0:
            if line[1] < line[5]:
                dir = -1
            else:
                dir = 1
        else:
            dir = dir = 1
    else:
        if line[5] > 0:
            if line[1] < line[5]:
                dir = 1
            else:
                dir = -1
        else:
            dir = dir = -1
    line[1] = abs(line[1])
    line.insert(4, dir)
    line = ",".join( [ str(i) for i in line ] )
    return line


def add_direction(path_file, new_path):
    f = sc.textFile(path_file)
    header = f.first()
    f = f.filter(lambda line: line!=header)
    f = f.map(add_direction_aux)
    f.repartition(1).saveAsTextFile(new_path)

def to_cart(line):
    x = line[4]*math.cos(line[5])
    y = line[4]*math.sin(line[5])
    line = line[:4]
    line.append(x)
    line.append(y)
    return line
    
create_training_testing_files("../dataset/360_leftArmMovement.csv", "./x", 0.001)

f = f.map(lambda line: [ line[0], line[1], line[2], line[3], line[4]*math.cos(line[5]), line[5] ])