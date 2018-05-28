"""Execute an typical training for a nn."""
# !/usr/bin/env
import sys

import math
import random
import re
import numpy as np
from defined_variables import *
sys.path.append("../classes/Log")
sys.path.append("../classes")
from PersistanceManager import PersistanceManager
from LogString import LogString
import Graphics as gf
import time

final_train_error = 0.0
final_test_error = 0.0
final_fulltest_error = 0.0
final_fulltest_stddev = 0.0

def summary_weights(w, n_inputs, n_nodes, name):
    for i in range(n_inputs):
        for j in range(n_nodes):
            pass#tf.summary.scalar("%d_%d" % (i,j), w[i][j])

def create_layer(inputs, n_inputs_nodes, n_nodes, name, 
                 activation_function=None, rate_prob=None, num_summaries=10):
    """
    Create a layer for a neural network.

    Args:
        inputs: The variable to multiply by the weights of the layer.
        n_inputs_nodes: Number of nodes of the inputs.
        n_nodes: Number of nodes for the new layer.
        name: The name of the scope.
        activation_function: If None than the layer does not have activation function else the activation function is this one.
        rate_prob: If None than the layer does not have dropout else it does, with this rate probability (% of input units to dropout)

    Return:
        w: Weight variable.
        b: Biases variable.
        o: New layer.
    """
    with tf.name_scope(name):
        w = tf.Variable(tf.random_normal(
            [n_inputs_nodes, n_nodes], 0.0, 0.1, tf.float32), name="weight")
        b = tf.Variable(tf.random_normal([n_nodes], 0.0, 0.1, tf.float32), name="biases")
        o = tf.add(tf.matmul(inputs, w), b)
        summary_weights(w, n_inputs, n_nodes, name)
        if activation_function is not None:
            o = activation_function(o)
        if rate_prob is not None:
            o = tf.layers.dropout(o, rate_prob,
                                training=training)
        return w, b, o


def create_model():
    """
    Create a new neuronal network.

    It uses the variables: n_inputs, hidden_layers_nodes, dropout_rate and
    n_outputs; to create each layer. Those are defined at the top of the file.
    Each variables that is created, is stored in a dictionary to be saved or 
    loaded later.
    Also uses the placeholders X and Y. X: inputs, Y:labels. They are defined 
    at the top of the file too.

    Return:
        layer: The output layer of the neuronal net.
        saved_variables: The variables to be saved.
    """
    saved_variables = {}

    w, b, layer = create_layer(X, n_inputs, hidden_layers_nodes[0], 
                                        "HL1", hidden_layers_ac_fun[0], dropout_rate[0])
    saved_variables[w.name] = w
    saved_variables[b.name] = b

    for i in range(1, len(hidden_layers_nodes)):
        w, b, layer = create_layer(
            layer, hidden_layers_nodes[i-1], hidden_layers_nodes[i],
            "HL"+str(i+1), hidden_layers_ac_fun[i], dropout_rate[i])
        saved_variables[w.name] = w
        saved_variables[b.name] = b

    w, b, layer = create_layer(
        layer, hidden_layers_nodes[-1], n_outputs, "output", tf.nn.sigmoid, None)
    saved_variables[w.name] = w
    saved_variables[b.name] = b
    return layer, saved_variables

prediction, saved_variables = create_model()

#mean_absolute_error
#cost_mse = tf.losses.mean_squared_error(labels=Y,
#                                                predictions=prediction)
cost_mse = tf.square( tf.subtract( Y, prediction ) )
optimizer = tf.train.GradientDescentOptimizer(
    learning_rate=learning_rate).minimize(cost_mse)
cost_rmse = tf.sqrt(tf.losses.mean_squared_error(labels=Y, predictions=prediction))

def train(sess, persistance_manager):
    """
    Execute an typical training.

    First loads the full training dataset and the testing dataset. Trains the neuronal net with
    the registers, and iterates the number of times that the variable
    iteration has. All the variables (batch_size, iterations)
    are defined at the top of the file.

    Args:
        sess: The tf.Session() where the training will be executed.
        saver: The one who saves the variables.
        ckpt: The next checkpoint of the saved variables.
    """
    global final_train_error, final_test_error
    #######
    summaries = []
    #merged_summary_op = tf.summary.merge_all()
    #summary_writer = tf.summary.FileWriter("./logs", graph=tf.get_default_graph())
    #######
    error_log = LogString(error_log_path, "w+", "iteration,train_error,test_error\n")

    for iteration in range(iterations):
        avg_cost_train = 0
        
        # Begin training
        training_dataset.restore_index()
        training_dataset.shuffle()
        sess.run(training_mode_op, feed_dict={mode: False})
        while(not training_dataset.dataset_out_of_range()):
            inputs, labels = training_dataset.get_next()
            _, cost_aux = sess.run(
                [optimizer, cost_mse], feed_dict={X: inputs, Y: labels})
            avg_cost_train += cost_aux
        #summarie = sess.run(merged_summary_op)
        #summaries.append(summarie)

        # Begin testing
        testing_dataset.restore_index()
        avg_cost_test = 0
        sess.run(training_mode_op, feed_dict={mode: False})
        while(not testing_dataset.dataset_out_of_range()):
            inputs, labels = testing_dataset.get_next()
            cost_aux = sess.run(cost_rmse,feed_dict={X: inputs, Y: labels})
            avg_cost_test += cost_aux
        train_error = avg_cost_train/training_dataset.get_size()
        test_error = avg_cost_test/testing_dataset.get_size()

        final_fulltest_error = test_error
        final_train_error = train_error

        print("Iteration: %d\ttrain cost: %f\ttest cost: %f" % (
               iteration, train_error, test_error))
        error_log.log_string([train_error[0][0], test_error])
    
    error_log.close_file()
    #for i in range(len(summaries)):
    #    summary_writer.add_summary( summaries[i], i )
    
    #persistance_manager.save_variables()


def incremental_train(sess, persistance_manager):
    #######
    summaries = []
    global final_test_error, final_train_error
    #merged_summary_op = tf.summary.merge_all()
    #summary_writer = tf.summary.FileWriter("./logs", graph=tf.get_default_graph())
    #######
    error_log = LogString(error_log_path, "w+", "iteration,train_error,test_error\n")
    iteration = 0
    while not training_increment_dataset.increment_out_of_range():
        avg_cost_train = 0
        
        # Begin training
        for _ in range(iterations):
            training_increment_dataset.restore_index()
            #training_increment_dataset.shuffle()
            sess.run(training_mode_op, feed_dict={mode: False})
            while(not training_increment_dataset.dataset_out_of_range()):
                inputs, labels = training_increment_dataset.get_next()
                _, cost_aux = sess.run(
                    [optimizer, cost_mse], feed_dict={X: inputs, Y: labels})
                
                avg_cost_train += cost_aux
        
            #summarie = sess.run(merged_summary_op)
            #summaries.append(summarie)
            # Begin testing
            testing_dataset.restore_index()
            avg_cost_test = 0

            sess.run(training_mode_op, feed_dict={mode: False})
            while(not testing_dataset.dataset_out_of_range()):
                inputs, labels = testing_dataset.get_next()
                cost_aux = sess.run(cost_rmse, feed_dict={X: inputs, Y: labels})
                avg_cost_test += cost_aux
            train_error = avg_cost_train/(training_increment_dataset.get_size()*iterations)
            test_error = avg_cost_test/testing_dataset.get_size()
            
            final_train_error = train_error
            final_test_error = test_error
            
            print("Iteration: %d\ttrain cost: %f\ttest cost: %f" % (
                iteration, train_error, test_error))
            iteration += 1
            error_log.log_string([train_error[0][0], test_error])
        training_increment_dataset.increment_dataset()
    
    error_log.close_file()
    #for i in range(len(summaries)):
    #    summary_writer.add_summary( summaries[i], i )
    
    #persistance_manager.save_variables()

def test(sess):
    """
    Tests the neuronal net with the full dataset.

    Calculates the root mean squared error.

    Args:
        sess: The tf.Session() where the testing will be executed.
    """
    global final_fulltest_error, final_fulltest_stddev
    avg_cost = 0
    avg_cost_squared = 0
    predictions_log = LogString(predictions_log_path, "w+", "iteration,prediction,label\n")
    sess.run(training_mode_op, feed_dict={mode: False})
    cont = 0
    while not full_dataset.dataset_out_of_range():
        inputs, labels = full_dataset.get_next()
        cost, predic = sess.run([cost_rmse, prediction], feed_dict={
                                    X: inputs, Y: labels})
        avg_cost += cost
        avg_cost_squared += cost**2
        predictions_log.log_string([predic[0][0], labels[0][0]])
        if cont % 1000 == 0:
            predictions_log.save()
        cont += 1
    avg_cost /= cont
    avg_cost_squared /= cont
    var = avg_cost_squared + avg_cost**2

    stddev = math.sqrt(var)
    final_fulltest_error = avg_cost
    final_fulltest_stddev = stddev
    predictions_log.close_file()
    print("Cost: %f\tStddev: %f" % (avg_cost, stddev))


def save_tests_logs():
    lines_number = 0
    f =  open("test_nn.csv", "r+")
    lines_number = len(f.readlines())
    string = "%d,%d,%d,%f,%f,%f,%f,%f\n"%(lines_number, memory_size, iterations, learning_rate, final_train_error, final_test_error, final_fulltest_error, final_fulltest_stddev)
    f.write(string)
    f.close()
    return lines_number

def main():
    """Execute program."""
    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        persistance_manager = PersistanceManager(sess, saved_variables, "./checkpoints/")  # For the checkpoints
        persistance_manager.load_variables()
        #train(sess, persistance_manager)
        #incremental_train(sess, persistance_manager)
        #test(sess)
        #img_number = save_tests_logs()
        #"C:/Users/Usuario/Desktop/ariel/Practica/pnodes/test_nn/%d_e.png"%(img_number)
        gf.plot_csv(error_log_path, ",", 0,[1, 2], "Iteration", "Value", "Error log", ["g-", "r-"], True, None)
        #gf.plot_csv(predictions_log_path, ",", 0, [1,2], "Iteration", "Value", "Predictions log", ["r+", "ko"], False, "C:/Users/Usuario/Desktop/ariel/Practica/pnodes/test_nn/%d_p.png"%(img_number))


if __name__ == '__main__':
    main()