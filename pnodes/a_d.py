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
import keyboard
import tensorflow as tf

final_train_error = 0.0
final_test_error = 0.0
final_fulltest_error = 0.0
final_fulltest_stddev = 0.0
train_type = ""

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

cost_mse = tf.losses.mean_squared_error(labels=Y, predictions=prediction)
cost_placeholder = tf.placeholder(tf.float32)
cost_variable = tf.Variable(0.0)
cost_variable_op = tf.assign(cost_variable, cost_placeholder)
optimizer = tf.train.AdamOptimizer(
    learning_rate=learning_rate, beta1=beta_1, beta2=beta_2, epsilon=epsilon).minimize(cost_mse)
#optimizer = tf.train.AdagradDAOptimizer(learning_rate, tf.cast(0, tf.int64)).minimize(cost_mse)
cost_rmse = tf.sqrt(tf.losses.mean_squared_error(labels=Y, predictions=prediction))


def batch_memory(memory):
    inputs = []
    labels = []
    data = []
    size = len(memory)
    for i in memory:
        inputs_aux, labels_aux = i
        inputs.append(inputs_aux[0])
        labels.append(labels_aux[0])
    if mini_batch_size == 0:
        memory = [inputs, labels]
    else:
        num_of_batches = size//mini_batch_size
        for i in range(num_of_batches):
            data.append( [ inputs[i*mini_batch_size:(i+1)*mini_batch_size], labels[i*mini_batch_size:(i+1)*mini_batch_size]  ] )
        if num_of_batches == 0:
            data.append( [ inputs[:mini_batch_size], labels[:mini_batch_size]  ] )
    return data



def stochastic_train_mini_batch(sess, persistance_manager):
    
    global final_test_error, final_train_error, train_type
    train_type = "Stochastic_Mini-batch"
    
    error_log = LogString(error_log_path, "w+", "train_step,train_error,test_error\n")
    time_log = LogString(time_log_path, "w+", "train_step,time\n")
    train_step_count = 0
    iteration = 0
    #sess.run(training_mode_op, feed_dict={mode: False})
    
    validation_dataset_size = len(validation_dataset)
    memory = []
    memory_len = 0
    for x_train, y_train in train_dataset:
        if keyboard.is_pressed('q'):
            break
        memory.append([x_train, y_train])
        memory_len += 1
        if memory_len > memory_size:
            memory = memory[1:]
            memory_len = memory_size

        total_time = 0
        for train_step in range(train_steps):
            if keyboard.is_pressed('q'):
                break
            avg_cost_train = 0
            sess.run(training_mode_op, feed_dict={mode: train_dropout})
            step_start_t = time.time()
            memory_aux = batch_memory(memory)
            for x,y in memory_aux:
                _, cost_aux = sess.run([optimizer, cost_mse], feed_dict={X: x, Y: y})
                avg_cost_train += cost_aux
            step_finish_t = time.time()
            avg_cost_test = 0
            sess.run(training_mode_op, feed_dict={mode: False})
            for x_validation, y_validation in validation_dataset:
                cost_aux = sess.run(cost_rmse, feed_dict={X: x_validation, Y: y_validation})
                avg_cost_test += cost_aux

            train_error = avg_cost_train/memory_len
            test_error = avg_cost_test/validation_dataset_size
            final_train_error = train_error
            final_test_error = test_error
            print("Train_step: %d\ttrain cost: %f\ttest cost: %f" % (
                train_step_count, train_error, test_error))
            train_step_count += 1
            error_log.log_string([train_error, test_error])
            total_time += (step_finish_t-step_start_t)
        if iteration != 0 and iteration % 100 == 0:
            time_log.save()
            error_log.save()
            #persistance_manager.save_variables()
        iteration += 1
        time_log.log_string([total_time])
    error_log.close_file()
    time_log.close_file()
    

def stochastic_train_memory(sess, persistance_manager):
    
    global final_test_error, final_train_error, train_type
    train_type = "Stochastic_Memory"
    
    error_log = LogString(error_log_path, "w+", "train_step,train_error,test_error\n")
    time_log = LogString(time_log_path, "w+", "train_step,time\n")
    train_step_count = 0
    iteration = 0
    #sess.run(training_mode_op, feed_dict={mode: False})
    
    validation_dataset_size = len(validation_dataset)
    memory = []
    memory_len = 0
    for x_train, y_train in train_dataset:
        if keyboard.is_pressed('q'):
            break
        memory.append([x_train, y_train])
        memory_len += 1
        if memory_len > memory_size:
            memory = memory[1:]
            memory_len = memory_size
        total_time = 0
        for train_step in range(train_steps):
            if keyboard.is_pressed('q'):
                break
            avg_cost_train = 0
            sess.run(training_mode_op, feed_dict={mode: train_dropout})
            step_start_t = time.time()
            for x,y in memory:
                _, cost_aux = sess.run([optimizer, cost_mse], feed_dict={X: x, Y: y})
                avg_cost_train += cost_aux
            step_finish_t = time.time()
            avg_cost_test = 0
            sess.run(training_mode_op, feed_dict={mode: False})
            for x_validation, y_validation in validation_dataset:
                cost_aux = sess.run(cost_rmse, feed_dict={X: x_validation, Y: y_validation})
                avg_cost_test += cost_aux

            train_error = avg_cost_train/memory_len
            test_error = avg_cost_test/validation_dataset_size
            final_train_error = train_error
            final_test_error = test_error
            print("Train_step: %d\ttrain cost: %f\ttest cost: %f" % (
                train_step_count, train_error, test_error))
            train_step_count += 1
            error_log.log_string([train_error, test_error])
            total_time += (step_finish_t-step_start_t)
        if iteration != 0 and iteration % 100 == 0:
            time_log.save()
            error_log.save()
            #persistance_manager.save_variables()
        iteration += 1
        time_log.log_string([total_time])
    error_log.close_file()
    time_log.close_file()


def stochastic_train(sess, persistance_manager):
    global final_test_error, final_train_error, train_type
    train_type = "Stochastic"
    
    error_log = LogString(error_log_path, "w+", "iteration,train_error,test_error\n")
    time_log = LogString(time_log_path, "w+", "iteration,time\n")
    train_step_count = 0
    iteration = 0
    #sess.run(training_mode_op, feed_dict={mode: False})
    
    validation_dataset_size = len(validation_dataset)
    for x_train, y_train in train_dataset:
        if keyboard.is_pressed('q'):
            break
        sess.run(training_mode_op, feed_dict={mode: train_dropout})
        iteration_start_t = time.time()
        avg_cost_train = 0
        for train_step in range(train_steps):
            _, cost_aux = sess.run([optimizer, cost_mse], feed_dict={X: x_train, Y: y_train})
            avg_cost_train += cost_aux
        iteration_finish_t = time.time()
        avg_cost_test = 0
        sess.run(training_mode_op, feed_dict={mode: False})
        for x_validation, y_validation in validation_dataset:
            cost_aux = sess.run(cost_rmse, feed_dict={X: x_validation, Y: y_validation})
            avg_cost_test += cost_aux

        train_error = avg_cost_train/train_steps
        test_error = avg_cost_test/validation_dataset_size
        final_train_error = train_error
        final_test_error = test_error
        print("Iteration: %d\ttrain cost: %f\ttest cost: %f" % (
            train_step_count, train_error, test_error))
        train_step_count += 1
        error_log.log_string([train_error, test_error])
        time_log.log_string([iteration_finish_t-iteration_start_t])
        if iteration != 0 and iteration % 100 == 0:
            time_log.save()
            error_log.save()
            #persistance_manager.save_variables()
        iteration += 1
    error_log.close_file()
    time_log.close_file()

def batch_dataset(size_of_batches):
    global train_dataset
    inputs = []
    labels = []
    size = len(train_dataset)
    for i in train_dataset:
        inputs_aux, labels_aux = i
        inputs.append(inputs_aux[0])
        labels.append(labels_aux[0])
    if size_of_batches == 0 or size_of_batches == 1:
        train_dataset = [inputs, labels]
    else:
        num_of_batches = size//size_of_batches
        data = []
        for i in range(num_of_batches):
            data.append( [ inputs[i*size_of_batches:(i+1)*size_of_batches], labels[i*size_of_batches:(i+1)*size_of_batches]  ] )
        train_dataset = data



def batch_train(sess, persistance_manager):
    global final_test_error, final_train_error, train_steps, train_type
    train_type = "Batch"
    batch_dataset(5)
    error_log = LogString(error_log_path, "w+", "iteration,train_error,test_error\n")
    time_log = LogString(time_log_path, "w+", "iteration,time\n")
    train_step_count = 0
    iteration = 0
    #sess.run(training_mode_op, feed_dict={mode: False})
    
    validation_dataset_size = len(validation_dataset)
    avg_cost_train = 0
    while not keyboard.is_pressed('q'):
        sess.run(training_mode_op, feed_dict={mode: train_dropout})
        iteration_start_t = time.time()
        avg_cost_train = 0
  
        for inputs, labels in train_dataset:

            train_error, _ = sess.run([cost_mse, optimizer], feed_dict={X: inputs, Y: labels})
            avg_cost_train += train_error
        avg_cost_train /= len(train_dataset)
        train_error = avg_cost_train

        iteration_finish_t = time.time()
        avg_cost_test = 0
        sess.run(training_mode_op, feed_dict={mode: False})
        for x_validation, y_validation in validation_dataset:
            cost_aux = sess.run(cost_rmse, feed_dict={X: x_validation, Y: y_validation})
            avg_cost_test += cost_aux
        test_error = avg_cost_test/validation_dataset_size
        final_train_error = train_error
        final_test_error = test_error
        print("Iteration: %d\ttrain cost: %f\ttest cost: %f" % (
            train_step_count, train_error, test_error))
        train_step_count += 1
        error_log.log_string([train_error, test_error])
        time_log.log_string([iteration_finish_t-iteration_start_t])
        if iteration != 0 and iteration % 100 == 0:
            time_log.save()
            error_log.save()
            #persistance_manager.save_variables()
        iteration += 1
    train_steps = iteration
    error_log.close_file()
    time_log.close_file()

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
    cont = len(test_dataset)
    for x_test, y_test in test_dataset:
        cost_aux, predic = sess.run([cost_rmse, prediction], feed_dict={
                                    X: x_test, Y: y_test})
        avg_cost += cost_aux
        avg_cost_squared += cost_aux**2
        predictions_log.log_string([predic[0][0], y_test[0][0]])
    avg_cost /= cont
    avg_cost_squared /= cont
    var = avg_cost_squared + avg_cost**2

    stddev = math.sqrt(var)
    final_fulltest_error = avg_cost
    final_fulltest_stddev = stddev
    predictions_log.close_file()
    print("Cost: %f\tStddev: %f" % (avg_cost, stddev))


def get_execution_number():
    f =  open("./tests/log.csv", "r+")
    lines_number = len(f.readlines())
    f.close()
    return lines_number

def save_tests_logs(exec_number, time):
    f =  open("./tests/log.csv", "a+")
    nodes_string = " ".join([str(i) for i in hidden_layers_nodes])
    ac_fun = " ".join(hidden_layers_ac_fun_names)
    dropout = []
    if not train_dropout:
        dropout.append("False")
    else:
        for i in dropout_rate:
            dropout.append(str(i))
    dropout = " ".join(dropout)
    optimizer_name = "%s %f %f %f"%(optimizer.name, beta_1, beta_2, epsilon)
    #optimizer_name = optimizer.name
    memory_size_str = "%d %d"%(memory_size, mini_batch_size)
    string = "%d,%s,%d,%f,%f,%f,%f,%s,%s,%s,%d,%f,%s,%s,%s\n"%(exec_number, memory_size_str, train_steps, learning_rate, final_train_error, final_test_error, final_fulltest_error, final_fulltest_stddev, optimizer_name,nodes_string,pnode_number,time,ac_fun,dropout, train_type)
    f.write(string)
    f.close()

def main():
    """Execute program."""
    global error_log_path, predictions_log_path, time_log_path
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        persistance_manager = PersistanceManager(sess, saved_variables, checkpoints_path)  # For the checkpoints
        persistance_manager.load_variables()
        exec_number = get_execution_number()
        error_log_path += "_%d.csv"%(exec_number)
        predictions_log_path += "_%d.csv"%(exec_number)
        time_log_path += "_%d.csv"%(exec_number)
        start_time = time.time()
        #stochastic_train(sess, persistance_manager)
        stochastic_train_mini_batch(sess, persistance_manager)
        #batch_train(sess, persistance_manager)
        finish_time = time.time()
        exec_time = finish_time-start_time
        print("Training time: %f"%(exec_time))
        test(sess)
        save_tests_logs(exec_number, exec_time)
        gf.plot_csv(error_log_path, ",", 0,[1, 2], "Iteration", "Value", "Error log", ["g.", "r."], True, "C:/Users/Usuario/Desktop/ariel/Practica/pnodes/tests/errors_images/%d_e.png"%(exec_number))
        gf.plot_csv(predictions_log_path, ",", 0, [1,2], "Iteration", "Value", "Predictions log", ["r+", "ko"], True, "C:/Users/Usuario/Desktop/ariel/Practica/pnodes/tests/predictions_images/%d_p.png"%(exec_number))
        gf.plot_csv(time_log_path, ",", 0, [1], "Iteration", "Seconds", "Times log", ["k."], False, "C:/Users/Usuario/Desktop/ariel/Practica/pnodes/tests/times_images/%d_t.png"%(exec_number))
        print(exec_number)
if __name__ == '__main__':
    main()