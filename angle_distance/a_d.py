"""Execute an typical training for a nn."""
# !/usr/bin/env
import math
import random
from WeightLog import WeightLog
import re
from PersistanceManager import PersistanceManager
from ErrorLog import ErrorLog
from defined_variables import *


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
        layer, hidden_layers_nodes[-1], n_outputs, "output", None, None)
    saved_variables[w.name] = w
    saved_variables[b.name] = b
    return layer, saved_variables

prediction, saved_variables = create_model()
cost_mse = tf.losses.mean_squared_error(labels=Y, predictions=prediction)
optimizer = tf.train.AdamOptimizer(
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
    
    #######
    log_weights_variables = []
    for i in saved_variables:
        if(re.match(".*weight.*", i)):  # Just weights, excludes biases
            log_weights_variables.append(saved_variables[i])
    #######

    weight_log = WeightLog(variables=log_weights_variables, sess=sess, 
                           log_path=weight_log_path, separator=",")
    error_log = ErrorLog(error_log_path, "w+")

    for iteration in range(iterations):
        avg_cost_train = 0
        
        # Begin training
        training_dataset.restore_index()
        training_dataset.shuffle()
        sess.run(training_mode_op, feed_dict={mode: True})
        while(not training_dataset.out_of_range()):
            inputs, labels = training_dataset.get_next()
            _, cost_aux = sess.run(
                [optimizer, cost_mse], feed_dict={X: inputs, Y: labels})
            avg_cost_train += cost_aux
        
        weight_log.log_weights()
        weight_log.save_log()
        
        # Begin testing
        testing_dataset.restore_index()
        avg_cost_test = 0
        sess.run(training_mode_op, feed_dict={mode: False})
        while(not testing_dataset.out_of_range()):
            inputs, labels = testing_dataset.get_next()
            cost_aux = sess.run(cost_rmse, feed_dict={X: inputs, Y: labels})
            avg_cost_test += cost_aux
        train_error = avg_cost_train/training_dataset.get_size()
        test_error = avg_cost_test/testing_dataset.get_size()
        print("Iteration: %d\ttrain cost: %f\ttest cost: %f" % (
               iteration, train_error, test_error))
        
        error_log.log_error(train_error, test_error)
    
    error_log.close_file()
    persistance_manager.save_variables()
    #weight_log.close_file()


def test(sess):
    """
    Tests the neuronal net with the full dataset.

    Calculates the root mean squared error.

    Args:
        sess: The tf.Session() where the testing will be executed.
    """
    
    avg_cost = 0

    predictions = open(predictions_log_path, "w+")
    sess.run(training_mode_op, feed_dict={mode: False})
    predictions_list = ["predict_x,predict_y,label_x,label_y\n"]
    while not full_dataset.out_of_range():
        inputs, labels = full_dataset.get_next()
        predic, cost_aux = sess.run([prediction, cost_rmse], feed_dict={
                                    X: inputs, Y: labels})
        predictions_list.append("%f,%f,%f,%f\n" % (predic[0][0], predic[0][1], 
                                labels[0][0], labels[0][1]))
        avg_cost += cost_aux
    predictions.write("".join(predictions_list))
    predictions.close()
    print("Full dataset avg cost rmse: %f" % (avg_cost/full_dataset.get_size()))


def main():
    """Execute program."""
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        persistance_manager = PersistanceManager(sess, saved_variables, "./checkpoints/")  # For the checkpoints
        persistance_manager.load_variables()
        train(sess, persistance_manager)
        test(sess)


if __name__ == '__main__':
    main()
