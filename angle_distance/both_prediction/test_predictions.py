import math

def cart_2_pol(x, y):
    r = math.sqrt( x**2 + y**2 )
    a = math.atan2( y , x )
    
    return r, a

def cart_2_pol_predictions():
    predictions = open("./predictions_log.csv", "r+")
    leftArmMovement = open("../datasets_angle_distance/leftArmMovement.csv", "r+")

    predictions_lines = predictions.readlines()[1:]
    leftArmMovement_lines = leftArmMovement.readlines()[1:]

    predictions.close()
    leftArmMovement.close()

    log = ["predition_distance;prediction_angle;correct_distance;correct_angle\n"]

    distance_difference = 0.0
    angle_difference = 0.0
    cont = 0
    for p,l in zip(predictions_lines, leftArmMovement_lines):
        p = [float(i) for i in p.split(",")]
        l = [float(i) for i in l.split(",")]
        
        r, a = cart_2_pol( p[1], p[2] )
        #if a > math.pi:
        #    a = a%math.pi - math.pi
        log.append("%f;%f;%f;%f\n"%(r, a, l[3], l[4]))
        if (  (a>0 and l[4]<0) or (a<0 and l[4]>0)  ):
            print(cont)
        cont += 1
        #print "(%f, %f) (%f, %f)"%(r, a, l[3], l[4])
        angle_difference += ( a -  l[4]) **2
        distance_difference += (r - l[3]) **2
    rmse = math.sqrt( (angle_difference+distance_difference)/(2*len(predictions_lines)) )
    print("RMSE: %f"%(rmse))
    with open( "real_predictions.csv", "w+") as log_file:
        log_file.write( "".join(log) )

def unnormalize_data(d, a):
    d *= 2.69
    a = math.pi*( 2*a - 1 )
    return d, a


def normalized_predictions():
    predictions = open("./predictions.csv", "r+")
    leftArmMovement = open("../dataset/leftArmMovement.csv", "r+")

    predictions_lines = predictions.readlines()[1:]
    leftArmMovement_lines = leftArmMovement.readlines()[1:]

    predictions.close()
    leftArmMovement.close()

    log = ["predition_distance;prediction_angle;correct_distance;correct_angle\n"]

    distance_difference = 0.0
    angle_difference = 0.0
    cont = 0
    for p,l in zip(predictions_lines, leftArmMovement_lines):
        p = [float(i) for i in p.split(",")]
        l = [float(i) for i in l.split(",")]
        
        d, a = unnormalize_data( p[0], p[1] )
        #if a > math.pi:
        #    a = a%math.pi - math.pi
        log.append("%f;%f;%f;%f\n"%(d, a, l[3], l[4]))
        if (  (a>0 and l[4]<0) or (a<0 and l[4]>0)  ):
            print(cont)
        cont += 1
        #print "(%f, %f) (%f, %f)"%(r, a, l[3], l[4])
        angle_difference += ( a -  l[4]) **2
        distance_difference += (d - l[3]) **2
    rmse = math.sqrt( (angle_difference+distance_difference)/(2*len(predictions_lines)) )
    print("RMSE: %f"%(rmse))
    with open( "real_predictions.csv", "w+") as log_file:
        log_file.write( "".join(log) )

cart_2_pol_predictions()