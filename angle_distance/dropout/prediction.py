import tensorflow as tf
import numpy as np
import random
import time

n_in = 3
n_out = 2

X = tf.placeholder(tf.float32, [None, n_in], name="Features")
Y = tf.placeholder(tf.float64, [None, n_out], name = "Labels")

learning_rate = 0.0001

def parseLine(line):
	    distance_t, angle_t, angle_act, distance_t1, angle_t1  = tf.decode_csv(line, [[0.0], [0.0] , [0.0], [0.0], [0.0]], field_delim=" ")
	    features = tf.stack([distance_t, angle_t, angle_act])
	    labels = tf.stack([distance_t1, angle_t1])
	    return features, labels

def createDataset(path):
	ds = tf.data.TextLineDataset(path)
	ds = ds.map(parseLine)
	return ds

def createLayer(inputs, inSize, size, activation, name, summary):
	weights = tf.get_variable(name="Weight", shape=[inSize, size], initializer=tf.random_normal_initializer(mean = 0, stddev = 0.1))
	biases = tf.get_variable(name="Biases", shape=[size], initializer=tf.random_normal_initializer(mean = 0, stddev = 0.1))
	if activation:
		with tf.name_scope("Activation"):
			return tf.nn.relu(tf.add(tf.matmul(inputs, weights, name="Multiply"), biases, name = "AddBiases"), name = "Activation")
	else:
		return tf.add(tf.matmul(inputs, weights, name = "Multiply"), biases, name = "AddBiases")


def model(inputs, dropout, summary):
	with tf.variable_scope("Model"):
		hl1 = 10
		hl2 = 5
		with tf.variable_scope("HiddenLayer1", reuse = tf.AUTO_REUSE):
			hiddenLayer1 = createLayer(inputs, 3, hl1, True, "1", summary)
			if dropout:
				hiddenLayer1 = tf.nn.dropout(hiddenLayer1, keep_prob=0.9)
		with tf.variable_scope("HiddenLayer2", reuse = tf.AUTO_REUSE):
			hiddenLayer2 = createLayer(hiddenLayer1, hl1, hl2, True, "2", summary)
			if dropout:
				hiddenLayer2 = tf.nn.dropout(hiddenLayer2, keep_prob=0.8)
		with tf.variable_scope("OutputLayer", reuse = tf.AUTO_REUSE):
			outputLayer = createLayer(hiddenLayer2, hl2, 2, False, "out", summary)
			return outputLayer

with tf.name_scope("TrainDataset"):
	with tf.name_scope("ReadingData"):
		batchSize = 10
		trainingDatasetSize = tf.placeholder(tf.int64)
		datasetTraining = createDataset("../dataset/leftArmMovement.txt") #Loads all training dataset
	with tf.name_scope("TakeRegisters"):
		datasetCopy = datasetTraining.take(trainingDatasetSize) #Takes N number of registers
		datasetCopy = datasetCopy.shuffle(trainingDatasetSize, reshuffle_each_iteration=True) #Shuffles taken registers
		datasetCopy = datasetCopy.batch(batchSize)

with tf.name_scope("TrainIterator"):
	with tf.name_scope("Create"):
		iteratorTraining = datasetCopy.make_initializable_iterator() #Creates a new iterator
	with tf.name_scope("GetNext"):
		featureTraining, labelTraining = iteratorTraining.get_next()
	with tf.name_scope("Reinit"):
		iteratorTrainingInitializer = iteratorTraining.make_initializer(datasetCopy)

with tf.name_scope("TestDataset"):
	with tf.name_scope("ReadingData"):
		datasetTesting = createDataset("../dataset/data_last_500.txt")
		datasetTesting = datasetTesting.batch(1)

with tf.name_scope("TestIterator"):
	with tf.name_scope("Create"):
		iteratorTesting = datasetTesting.make_initializable_iterator()
	with tf.name_scope("GetNext"):
		featureTesting, labelTesting = iteratorTesting.get_next() 
	with tf.name_scope("Reinit"):
		iteratorTestingInitializer = iteratorTesting.make_initializer(datasetTesting)

with tf.name_scope("Testing"):
	predictionTesting = model(featureTesting, dropout = False, summary=False)
	with tf.name_scope("Cost"):
		costTesting = tf.losses.mean_squared_error(labels = labelTesting, predictions = predictionTesting) #ERROR CUADRATICO MEDIO


with tf.name_scope("Training"):
	predictionTraining = model(featureTraining, dropout=True, summary=False)
	with tf.name_scope("Cost"):
		costTraining = tf.losses.mean_squared_error(labels = labelTraining, predictions = predictionTraining) #ERROR CUADRATICO MEDIO
	with tf.name_scope("Optimizer"):
		optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(costTraining)



def trainAndTest(sess):
	start = time.time()

	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver()
	writer = tf.summary.FileWriter("./log/", sess.graph)

	ckpt = 0 ##Lleva el contador de los checkpoints

	try:
		lastCheckpoint = tf.train.latest_checkpoint('./checkpoints/')
		saver.restore(sess, lastCheckpoint)
		ckpt = int(lastCheckpoint.split("-")[-1])+1
		print "Weights successfully loaded!"
	except:
		saver.save(sess, "./checkpoints/model.ckpt", global_step=0)
		print "Could not load the weights"

	file = open("./LOG.csv", "a+")

	dataBuffer=""
	for train in range(1, 200):
		size = train*10
		
		sess.run(iteratorTrainingInitializer, feed_dict={trainingDatasetSize:size}) #Inicialize el iterador con size numero de registros
		sess.run(iteratorTestingInitializer)

		costAvg = 0
		iterations = 0
		while True:
			try:
				_, costAux = sess.run([optimizer, costTraining])
				costAvg+=costAux
				iterations += 1
			except tf.errors.OutOfRangeError:
				dataBuffer+=str(train*10)+";"+str(costAvg/(iterations*batchSize))+";"
				if train %10 == 0:
					saver.save(sess, "./checkpoints/model.ckpt", global_step=ckpt)
					ckpt += 1
				break
		costAvg = 0
		iterations = 0
		
		while True:
			try:
				costAux = sess.run(costTesting)
				costAvg+=costAux
				iterations += 1
			except tf.errors.OutOfRangeError:
				finalCost = costAvg/iterations
				dataBuffer+=str(finalCost)+"\n"
				print "Registers on training: "+str(train*10)+"\t\tTesting error: "+str(finalCost)
				break
		if(train%10 == 0):
			file.write(dataBuffer)
			dataBuffer = ""
		
	saver.save(sess, "./checkpoints/model.ckpt", global_step = ckpt)
	if len(dataBuffer) != 0:
		file.write(dataBuffer)
	file.close()
	endtime = time.time()
	print str(endtime-start)

config = tf.ConfigProto()
config.intra_op_parallelism_threads = 2
config.inter_op_parallelism_threads = 2

with tf.Session(config = config) as sess:
	trainAndTest(sess)



def main():
	graph = tf.Graph()
	with graph.as_default():
		pass