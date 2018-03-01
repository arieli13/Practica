import tensorflow as tf
import numpy as np
import random
import sys
#sys.path.append('../') #Now we can import NeuralNetwork
sys.path.append('./')
from NeuralNetwork import NeuralNetwork

#######################################
train_X = []
train_Y = []

test_X = []
test_Y = []

h_l = [256]

n_in = 3
n_out = 2

iterations = 1
batchSize = 10

X = tf.placeholder(tf.float64, [None, n_in])
Y = tf.placeholder(tf.float64, [None, n_out])

learning_rate = 0.0001

totalWeights = (n_in+1)*h_l[0]
for i in range(1, len(h_l)):
    totalWeights += (h_l[i-1]+1)*h_l[i]
totalWeights += (h_l[len(h_l)-1]+1)*n_out



file = open("./dataset/data_190.txt", "r")
for line in file:
    line = line.split(" ")
    line = map(float, line)
    train_X.append(np.array(line[:3]))
    train_Y.append(np.array(line[3:]))


file = open("./dataset/test_497.txt", "r")
for line in file:
    line = line.split(" ")
    line = map(float, line)
    test_X.append(np.array(line[:3]))
    test_Y.append(np.array(line[3:]))

def createLayer(n_weightsIn, n_weights, activationIn, name, activation = True):
    with tf.name_scope(name):
        weights = tf.Variable(tf.random_normal([n_weightsIn, n_weights], 0, 0.1,  dtype = tf.float64), name="Weights")
        tf.summary.histogram("Weights", weights)
        tf.summary.scalar("Media_Weights", tf.reduce_mean(weights))
        biases_hl = tf.Variable(tf.random_normal([n_weights], 0, 0.1, dtype = tf.float64), name="Biases")
        #tf.summary.histogram("Biases", biases_hl)
        if activation:
            activacion_hl = tf.nn.relu(tf.add(tf.matmul(activationIn, weights), biases_hl))
            #tf.summary.histogram("Activation_function", activacion_hl)
            return activacion_hl
        return tf.add(tf.matmul(activationIn, weights), biases_hl)

def model():
    with tf.name_scope("model"):
        lastLayer = createLayer(n_in, h_l[0], X, "Hidden_layer_1")
        for i in range(1, len(h_l)):
            lastLayer = createLayer(h_l[i-1], h_l[i], lastLayer, "Hidden_layer_"+str(i+1))
        outputLayer = createLayer(h_l[len(h_l)-1], n_out, lastLayer, "Output_layer", False) 
        return outputLayer


def initTraining(prediction):
    with tf.name_scope("cost"):
        cost = tf.losses.mean_squared_error(labels = Y, predictions = prediction) #ERROR CUADRATICO MEDIO
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
        tf.summary.scalar("cost", cost)
        return cost, optimizer

def printTest(respuesta, batch_y):
    print("           \tDistance\tAngle")
    print("Prediction:\t%8.6f\t%8.6f" % (respuesta[0][0], respuesta[0][1]))
    print("Label:     \t%8.6f\t%8.6f" % (batch_y[0][0], batch_y[0][1]))
    print("")

def saveData(path, data):
    with open(path, "a") as file:
        file.write(data)
        file.close()

def execute():
    prediction = model()
    cost, optimizer = initTraining(prediction) 
    nn = NeuralNetwork(prediction, cost, optimizer, "./angle_distance/model/angle&distance.ckpt", X, Y, "./angle_distance/logs/")
    #nn.loadWeights()
    trainAvgCost, trainTime = nn.train(train_X, train_Y, batchSize, iterations, log = True )
    #nn.saveWeights()
    testAvgCost, testTime = nn.test(test_X, test_Y)
    csv = "%8.6f;%f;%s;%d;%d;%d;%d;%d;%f;%f;%f\n"%(testAvgCost, trainAvgCost, str(h_l), totalWeights, iterations, batchSize, len(train_X), len(test_X), learning_rate, trainTime, testTime)
    nn.closeSession()
    saveData("./angle_distance/testLog.csv", csv)


execute()



















"""
def cargarmodel(sess):
    saver = tf.train.Saver()
    try:
        saver.restore(sess,rutaWeights)
        print("Weights cargados correctamente\n")
    except:
        print("No se pudieron cargar los Weights")

def guardarmodel(sess):
    saver = tf.train.Saver()
    saver.save(sess, rutaWeights)
    print("model salvado en: "+rutaWeights+"\n")

def entrenar():
    prediction = model()
    cost, optimizer = initTraining(prediction)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        cargarmodel(sess)
        
        ultimo_error_prom = 0.0

        for epoch in range(epochsEntrenamiento):
            costPromedio = 0.0
            batchTotal = len(train_X)//batchSize
            
            for i in range(batchTotal):

                indiceMenor = i*(batchSize)
                indiceMayor = indiceMenor+(batchSize)

                batch_x = train_X[indiceMenor : indiceMayor]
                batch_y = train_Y[indiceMenor : indiceMayor]
                _, pcost = sess.run([optimizer, cost], feed_dict={X: batch_x, Y: batch_y})
                costPromedio += pcost / batchTotal

            print("Epoch: "+str(epoch+1) + "\tcost: "+str(costPromedio))
            ultimo_error_prom = costPromedio
        
        guardarmodel(sess)

def testear():
    prediction = model()
    cost, _ = initTraining(prediction)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        cargarmodel(sess)
        costPromedio = 0

        for i in range(len(test_X)):
            batch_x = [test_X[i]]
            batch_y = [test_labels[i]]

            respuesta = sess.run(prediction, feed_dict={X: batch_x})
            costPromedio += sess.run(cost, feed_dict={X: batch_x, Y:batch_y}) 
            if(mostrarRegistrosTest):
                print str(respuesta[0][0]) + "/" +str(batch_y[0][0]) + "/"+ str(abs((float(batch_y[0][0] - respuesta[0][0])))) 
        costPromedio /= len(test_X)
        print ("ECM en test: \t\t"+ str(costPromedio))
        print ("Tamannio del batch: \t"+str(batchSize))
        sess.close()
"""