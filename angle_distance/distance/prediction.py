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

h_l = [32, 16]

n_in = 3
n_out = 1

iterations = 1000
batchSize = 20

X = tf.placeholder(tf.float64, [None, n_in])
Y = tf.placeholder(tf.float64, [None, n_out])

learning_rate = 0.0001

#Calculates the total number of weights that the nn has.
totalWeights = (n_in+1)*h_l[0] 
for i in range(1, len(h_l)):
    totalWeights += (h_l[i-1]+1)*h_l[i]
totalWeights += (h_l[len(h_l)-1]+1)*n_out


#Inits the traning data
file = open("./dataset/data_190.txt", "r")
for line in file:
    line = line.split(" ")
    line = map(float, line)
    train_X.append(np.array(line[:3]))
    train_Y.append(np.array([line[3]]))

#Inits the testing data
file = open("./dataset/test_497.txt", "r")
for line in file:
    line = line.split(" ")
    line = map(float, line)
    test_X.append(np.array(line[:3]))
    test_Y.append(np.array([line[3]]))

#Creates a single layer with RELU activation func.
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

#Creates the nn.
def model():
    with tf.name_scope("model"):
        lastLayer = createLayer(n_in, h_l[0], X, "Hidden_layer_1")
        for i in range(1, len(h_l)):
            lastLayer = createLayer(h_l[i-1], h_l[i], lastLayer, "Hidden_layer_"+str(i+1))
        outputLayer = createLayer(h_l[len(h_l)-1], n_out, lastLayer, "Output_layer", False) 
        return outputLayer

#Defines the cost function and the optimizer for the nn. In this case AdamOptimizer.
def initTraining(prediction):
    with tf.name_scope("cost"):
        cost = tf.losses.mean_squared_error(labels = Y, predictions = prediction) #ERROR CUADRATICO MEDIO
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
        tf.summary.scalar("cost", cost)
        return cost, optimizer

#Function passed over parameter to nn.test, so it can print the testing values.
def printTest(respuesta, batch_y):
    print("Prediction:\t%8.6f" % (respuesta[0][0]))
    print("Label:     \t%8.6f" % (batch_y[0][0]))
    print("")

#Write data on specific path.
def saveData(path, data):
    with open(path, "a") as file:
        file.write(data)
        file.close()

#Executes the training and testing of the nn.
def execute():
    prediction = model()
    cost, optimizer = initTraining(prediction) 
    nn = NeuralNetwork(prediction, cost, optimizer, "./distance/model/distance_distance.ckpt", X, Y, "./distance/logs/")
    #nn.loadWeights()
    trainAvgCost, trainTime = nn.train(train_X, train_Y, batchSize, iterations, log = True )
    #nn.saveWeights()
    testAvgCost, testTime = nn.test(test_X, test_Y, printTest)
    csv = "%8.6f;%f;%s;%d;%d;%d;%d;%d;%f;%f;%f\n"%(testAvgCost, trainAvgCost, str(h_l), totalWeights, iterations, batchSize, len(train_X), len(test_X), learning_rate, trainTime, testTime)
    nn.closeSession()
    saveData("./distance/testLog.csv", csv)


execute()
