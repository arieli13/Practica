import math
#from pyspark import SparkContext, SparkConf
#conf = SparkConf().setAppName("normalize_data")
#sc = SparkContext(conf=conf)


def pol_2_cart(line):
    try:
        r = line[0]
        ang = line[1]
        rp = 0.05
        angp = line[2]

        rf = line[3]  # Para assert
        angf = line[4]  # Para assert

        x = r*math.cos(ang)
        y = r*math.sin(ang)

        xp = rp*math.cos(angp)
        yp = rp*math.sin(angp)

        xf = rf*math.cos(angf)  # Para assert
        yf = rf*math.sin(angf)  # Para assert

        assert round(x-xp, 5) == round(xf, 5) and round(y-yp, 5) == round(yf, 5), "(%f,%f)  (%f, %f)"%(round(x-xp, 5), round(xf, 5) , round(y-yp, 5) , round(yf, 5))
        new_line = [ x, y, xp, yp, xf, yf ]
        new_line = [str(i) for i in new_line]
        return new_line
    except Exception as e:
        print e.message
        exit(0)



#lam = sc.textFile(".\\datasets\\leftArmMovement.txt")
#lam = sc.textFile("C:/Users/Usuario/Desktop/docker/Practica/python-neat/angle_distance/datasets/leftArmMovement.txt")
#lam = lam.map(lambda line: [float(i) for i in line.split(" ")])
#lam = lam.map( pol_2_cart  )
#x = x.map(lambda line: ' '.join(line))
#x.repartition(1).saveAsTextFile("./datasets/cart_coords.txt")
