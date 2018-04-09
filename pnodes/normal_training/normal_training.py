"""Execute an typical training for a nn."""
# !/usr/bin/env

import tensorflow as tf
import math
import random

dataset_train_path = "../datasets/normalizado/pnode05_03000_train.txt"
dataset_test_path = "../datasets/normalizado/pnode05_03000_test.txt"
dataset_full_path = "../datasets/normalizado/pnode05_03000.txt"

logs_dir = "./logs/"
##################
n_inputs = 8
n_outputs = 1
hidden_layers_nodes = [20, 20, 20]
dropout_rate = [0.1, 0.1, 0.1]

learning_rate = 0.001

batch_size = 10
iterations = 200
##################
training = tf.Variable(True)
mode = tf.placeholder(tf.bool)
training_mode_op = tf.assign(training, mode)

X = tf.placeholder(tf.float32, [None, n_inputs])
Y = tf.placeholder(tf.float32, [None, n_outputs])


def load_dataset(path):
    """
    Create a dataset from a csv.

    The path is of an existing csv, each line separated by carriage return and each column separeted by blank space. Nine cols.

    Args:
        path: the path of the csv.

    Return:
        dataset: A list. Structure: [ [ [input], [label] ], [[input], [label]] ]
    """
    dataset = []
    with open(path) as f:
        lines = f.readlines()
        for line in lines:
            line = line.split(" ")
            feature = line[:8]
            label = line[8:]
            feature = [float(i) for i in feature]
            label = [float(i) for i in label]
            dataset.append([feature, label])
    return dataset


def create_matrix_summary(matrix, n_inputs, n_nodes, max_summaries=10):
    """Create a tf.summary.scalar of each weight in the matrix(layer)...

    Args:
        matrix: The weights tensor object.
        n_nodes: Number of weights that the matrix has.
        n_inputs: Number of input neurones that the matrix has.
        max_summaries: The number of nodes to save summaries.

    Returns:
        A list of all the tf.summary.scalar of each weight in the layer.

    """
    summary = []
    for i in range(n_inputs):
        for j in range(n_nodes):
            summary.append(tf.summary.scalar("%d,%d" % (i, j), matrix[i][j]))
    if max_summaries < len(summary):
        summary = random.sample(summary, max_summaries)
    return summary


def create_layer(inputs, n_inputs_nodes, n_nodes, name, activation_function=None,
                 rate_prob=None, num_summaries=10):
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
        summary: The summary of the weights
    """
    with tf.name_scope(name):
        w = tf.Variable(tf.random_normal(
            [n_inputs_nodes, n_nodes], 0.0, 0.1, tf.float32))
        summary = create_matrix_summary(w, n_inputs_nodes, n_nodes)
        b = tf.Variable(tf.random_normal([n_nodes], 0.0, 0.1, tf.float32))
        o = tf.add(tf.matmul(inputs, w), b)
        if activation_function is not None:
            o = activation_function(o)
        if rate_prob is not None:
            o = tf.layers.dropout(o, rate_prob,
                                training=training)
        return w, b, o, summary


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
        summary_variables: Variables to save the summary of them.
    """
    saved_variables = {}
    summary_variables = []

    w, b, layer, summary = create_layer(X, n_inputs, hidden_layers_nodes[0], 
                                        "HL1", tf.nn.relu, dropout_rate[0])
    summary_variables += summary
    saved_variables[w.name] = w
    saved_variables[b.name] = b

    for i in range(1, len(hidden_layers_nodes)):
        w, b, layer, summary = create_layer(
            layer, hidden_layers_nodes[i-1], hidden_layers_nodes[i],
            "HL"+str(i+1), tf.nn.relu, dropout_rate[i])
        summary_variables += summary
        saved_variables[w.name] = w
        saved_variables[b.name] = b

    w, b, layer, summary = create_layer(
        layer, hidden_layers_nodes[-1], n_outputs, "output", tf.nn.sigmoid, None)
    summary_variables += summary
    saved_variables[w.name] = w
    saved_variables[b.name] = b
    return layer, saved_variables, summary_variables


def save_model(sess, saver, path, ckpt):
    """Save the current session...

    Args:
        sess: The session where is the model that will be saved.
        saver: An instance of tf.train.Saver().
        path: Path where the model will be saved.
        ckpt: The global_step of the checkpoint.

    """
    saver.save(sess, path, global_step=ckpt)  # Saves the weights of the model


def load_model(sess, saver, path):
    """Load the current session from a saved one...

    Args:
        sess: The current tensorflow session.
        saver: An instance of tf.train.Saver().
        path: The path where the model is saved.

    Return:
        ckpt: The next checkpoint number to save(global_step). If no saved model is found, returns 0.

    """
    ckpt = 0
    try:
        lastCheckpoint = tf.train.latest_checkpoint(path)
        saver.restore(sess, lastCheckpoint)
        ckpt = int(lastCheckpoint.split("-")[-1])+1
        print "Model successfully loaded"
    except:
        print "Could not load the model"
    finally:
        return ckpt


prediction, saved_variables, summary_variables = create_model()
cost_mse = tf.losses.mean_squared_error(labels=Y, predictions=prediction)
optimizer = tf.train.AdamOptimizer(
    learning_rate=learning_rate).minimize(cost_mse)

_, cost_rmse = tf.metrics.root_mean_squared_error(Y, prediction)


def train(sess, saver, ckpt):
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
    training_dataset = load_dataset(dataset_train_path)
    testing_dataset = load_dataset(dataset_test_path)

    training_dataset_size = len(training_dataset)
    testing_dataset_size = len(testing_dataset)

    assert training_dataset_size >= batch_size

    total_epochs = int(
        math.ceil(float(training_dataset_size)/float(batch_size)))

    log = open("LOG.csv", "w+")

    global summary_variables
    summary_variables = tf.summary.merge(summary_variables)
    file_writer = tf.summary.FileWriter(logs_dir, sess.graph)
    summary_number = 0
    for iteration in range(iterations):
        # iteration = 1
        # while True:
        min_index = 0
        avg_cost_train = 0
        random.shuffle(training_dataset)
        # Begin training
        sess.run(training_mode_op, feed_dict={mode: True})
        for _ in range(total_epochs):
            data = training_dataset[min_index:(min_index+batch_size)]
            x = [i[0] for i in data]
            y = [i[1] for i in data]
            min_index += batch_size
            _, cost_aux, summaries = sess.run(
                [optimizer, cost_mse, summary_variables], feed_dict={X: x, Y: y})
            avg_cost_train += cost_aux

        # Begin testing
        avg_cost_test = 0
        sess.run(training_mode_op, feed_dict={mode: False})
        for i in range(testing_dataset_size):
            data = testing_dataset[i:i+1]
            x = [i[0] for i in data]
            y = [i[1] for i in data]
            min_index += batch_size
            cost_aux = sess.run(cost_rmse, feed_dict={X: x, Y: y})
            avg_cost_test += cost_aux
        print "Iteration: %d\ttrain cost: %f\ttest cost: %f" % (
            iteration, avg_cost_train/training_dataset_size,
            avg_cost_test/testing_dataset_size)
        ckpt += 1
        save_model(sess, saver, "./checkpoints/", ckpt)
        log.write("%d;%f;%f\n" % (iteration+1, avg_cost_train /
                                  training_dataset_size,
                                  avg_cost_test/testing_dataset_size))
        file_writer.add_summary(summaries, summary_number)
        summary_number += 1


def test(sess):
    """
    Tests the neuronal net with the full dataset.

    Calculates the root mean squared error.

    Args:
        sess: The tf.Session() where the testing will be executed.
    """
    full_dataset = load_dataset(dataset_full_path)
    full_dataset_size = len(full_dataset)
    avg_cost = 0

    predictions = open("predictions.csv", "w+")
    sess.run(training_mode_op, feed_dict={mode: False})
    for data in full_dataset:
        predic, cost_aux = sess.run([prediction, cost_rmse], feed_dict={
                                    X: [data[0]], Y: [data[1]]})
        predictions.write("%f;%f\n" % (predic[0][0], data[1][0]))
        avg_cost += cost_aux
    print "Full dataset avg cost rmse: %f" % (avg_cost/full_dataset_size)


def main():
    """Execute program."""
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        saver = tf.train.Saver(saved_variables)
        ckpt = load_model(sess, saver, "./checkpoints/")
        train(sess, saver, ckpt)
        test(sess)


if __name__ == '__main__':
    main()
