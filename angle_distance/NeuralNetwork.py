import tensorflow as tf
import time
import numpy as np

class NeuralNetwork:

    def __init__(self, prediction, cost, optimazer, savePath, inPlaceholder, outPlaceholder, logPathTrain, logPathTest):
        self.prediction = prediction
        self.cost = cost
        self.optimazer = optimazer
        self.savePath = savePath
        self.inPlaceholder = inPlaceholder
        self.outPlaceholder = outPlaceholder
        self.logPathTrain = logPathTrain
        self.logPathTest = logPathTest
        self.sess = tf.Session()
        self.merged = tf.summary.merge_all()
        self.fileWriterTrain = tf.summary.FileWriter(logPathTrain,self.sess.graph)
        self.fileWriterTest = tf.summary.FileWriter(logPathTest, self.sess.graph)
        self.initSession()

    def initSession(self):
        self.sess.run(tf.global_variables_initializer())

    def closeSession(self):
        self.sess.close()

    def loadWeights(self, path = None): #Restores a saved session.
        try:
            if(path == None):
                path = self.savePath
            saver = tf.train.Saver()
            saver.restore(self.sess, path)
            print("Weights successfully loaded\n")
        except:
            print("Couldn't load the weights")

    def saveWeights(self, path = None): #Saves the current session.
        if(path == None):
            path = self.savePath
        saver = tf.train.Saver()
        saver.save(self.sess, path)
        print("Model saved in: "+path+"\n")

    def train(self, data, labels, batchSize = 1, iterations = 100, log = False, logPeriod = 1):
        startTime = time.time()
        globalAverageCost = 0
        logged = False
        for epoch in range(iterations):
            avgCost = 0.0
            batchTotal = len(data)//batchSize
            for i in range(batchTotal):

                minIndex = i*(batchSize)
                maxIndex = minIndex+(batchSize)

                batch_x = data[minIndex : maxIndex]
                batch_y = labels[minIndex : maxIndex]
                if(log and epoch%logPeriod == 0 and not logged):
                    merged, _, pcost = self.sess.run([self.merged, self.optimazer, self.cost], feed_dict={self.inPlaceholder: batch_x, self.outPlaceholder: batch_y})
                    self.fileWriterTrain.add_summary(merged, epoch)
                    logged = True
                else:
                    _, pcost = self.sess.run([self.optimazer, self.cost], feed_dict={self.inPlaceholder: batch_x, self.outPlaceholder: batch_y})
                
                avgCost += pcost / batchTotal
                globalAverageCost = avgCost

            print("Epoch: "+str(epoch+1) + "\taverage cost: "+str(avgCost))
            logged = False
        endTime = time.time()
        totalTime = endTime - startTime
        return globalAverageCost, totalTime


    def test(self, data, labels, printFunction = None, test_summary = None):
        startTime = time.time()
        avgCost = 0
        dataSize = len(data)
        for i in range(dataSize):
            batch_x = [data[i]]
            batch_y = [labels[i]]

            test_summaryAux, pred, avgCostAux = self.sess.run([test_summary, self.prediction, self.cost], feed_dict={self.inPlaceholder: batch_x, self.outPlaceholder: batch_y})
            if(test_summary!=None):
                self.fileWriterTest.add_summary(test_summaryAux, i)
            avgCost += avgCostAux
            if(printFunction != None):
                printFunction(pred, batch_y)

        endTime = time.time()
        totalTime = endTime - startTime
        avgCost /= dataSize
        print ("\nAverage cost: \t"+ str(avgCost))
        return avgCost, totalTime

    def getCost(self, data, labels): #Gets the cost of some data.
        pcost = self.sess.run( self.cost, feed_dict={self.inPlaceholder: data, self.outPlaceholder:labels} )
        return pcost

    def predict(self, data): #Predicts value(s) with data as input.
        pred = self.sess.run(self.prediction, feed_dict={self.inPlaceholder:data})
        return pred

    def execute(self, function, data): #Executes any function in current Session
        return self.sess.run(function, feed_dict= data)
