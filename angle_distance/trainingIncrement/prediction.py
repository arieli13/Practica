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

h_l =             [64, 64]
summary_weights = [64, 64, 1] #La ultima es la del output

n_in = 3
n_out = 2

iterations = 200
batchSize = 10

X = tf.placeholder(tf.float64, [None, n_in])
Y = tf.placeholder(tf.float64, [None, n_out])

learning_rate = 0.001

#Calculates the total number of weights that the nn has.
totalWeights = (n_in+1)*h_l[0]
for i in range(1, len(h_l)):
    totalWeights += (h_l[i-1]+1)*h_l[i]
totalWeights += (h_l[len(h_l)-1]+1)*n_out


#Inits the traning data
file = open("./dataset/data_50.txt", "r")
for line in file:
    line = line.split(" ")
    line = map(float, line)
    train_X.append(np.array(line[:3]))
    train_Y.append(np.array(line[3:]))

#Inits the testing data
file = open("./dataset/test_497.txt", "r")
for line in file:
    line = line.split(" ")
    line = map(float, line)
    test_X.append(np.array(line[:3]))
    test_Y.append(np.array(line[3:]))

#Creates a single layer with RELU activation func.
def createLayer(n_weightsIn, n_weights, activationIn, name, activation = True, summary_weights_number = 1):
    with tf.name_scope(name+"_Statistics"):
        weights = tf.Variable(tf.random_normal([n_weightsIn, n_weights], 0, 0.1,  dtype = tf.float64), name="Weights")
        mean = tf.reduce_mean(weights)
        stddev = tf.sqrt(tf.reduce_mean(weights - mean))
        tf.summary.histogram("Weights", weights)
        tf.summary.scalar("Mean", mean)
        tf.summary.scalar("Stddev", mean)
        tf.summary.scalar("Max", tf.reduce_max(weights))
        tf.summary.scalar("Min", tf.reduce_min(weights))
    with tf.name_scope(name+"_EachWeight"):
        randomNumbers = random.sample(range(0, n_weights), summary_weights_number)
        for num in randomNumbers:
            tf.summary.scalar(str(num), weights[0][num])
        biases_hl = tf.Variable(tf.random_normal([n_weights], 0, 0.1, dtype = tf.float64), name="Biases")
        if activation:
            activacion_hl = tf.nn.relu(tf.add(tf.matmul(activationIn, weights), biases_hl))
            return activacion_hl
        return tf.add(tf.matmul(activationIn, weights), biases_hl)

#Creates the nn.
def model():
    lastLayer = createLayer(n_in, h_l[0], X, "Hidden_layer_1", summary_weights_number = summary_weights[0])
    for i in range(1, len(h_l)):
        lastLayer = createLayer(h_l[i-1], h_l[i], lastLayer, "Hidden_layer_"+str(i+1), summary_weights_number = summary_weights[i])
    outputLayer = createLayer(h_l[len(h_l)-1], n_out, lastLayer, "Output_layer", False, summary_weights_number = summary_weights[len(summary_weights)-1]) 
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
    print("           \tDistance\tAngle")
    print("Prediction:\t%8.6f\t%8.6f" % (respuesta[0][0], respuesta[0][1]))
    print("Label:     \t%8.6f\t%8.6f" % (batch_y[0][0], batch_y[0][1]))
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
    nn = NeuralNetwork(prediction, cost, optimizer, "./trainingIncrement/model/trainingInc.ckpt", X, Y, "./trainingIncrement/logs/train", "./trainingIncrement/logs/test")
    #nn.loadWeights()
    trainAvgCost, trainTime = nn.train(train_X, train_Y, batchSize, iterations, log = True, logPeriod = 1 )
    nn.saveWeights()
    accuracy = tf.summary.scalar("Accuracy_Test", cost)
    testAvgCost, testTime = nn.test(test_X, test_Y, printTest, accuracy)
    csv = "%8.6f;%f;%s;%d;%d;%d;%d;%d;%f;%f;%f\n"%(testAvgCost, trainAvgCost, str(h_l), totalWeights, iterations, batchSize, len(train_X), len(test_X), learning_rate, trainTime, testTime)
    nn.closeSession()
    saveData("./trainingIncrement/testLog.csv", csv)


execute()