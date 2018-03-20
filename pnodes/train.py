"""Creates the dataset for the training, cost, optimization and trains a neural network..."""

import tensorflow as tf
from dataset import create_dataset, create_iterator
from model import create_model


def create_training_dataset(batch_size, increment_dataset):
    """Create the dataset of training...
    
    It creates a variable named dataset_size, it is the number
        of registers to be taken each training. An operation is defined to increment 10 times the dataset_size, 
        so in each new training, the dataset_size increments N times and the number of registers to be taken will
        be greater...

    Args:
        batch_size: Size of the batch of the training dataset.
        increment_dataset: The number of registers to increment each training. 
            Ex: First train: 10 registers. Next train: 20 registers. increment_dataset = 10

    Returns:
        A dictionary with the variables:
            dataset: The dataset for training.
            dataset_resize_op: The resize operation to increment the number of registers to take each training.

    """
    dataset_size = tf.Variable(0, dtype=tf.int64)
    dataset_resize_op = dataset_size.assign(
        tf.add(dataset_size, increment_dataset))
    complete_dataset = create_dataset(
        "./datasets/normalizado/pnode01_03000_train.txt")  # Loads all training dataset
    trainable_dataset = complete_dataset.take(dataset_size)
    shuffled_dataset = trainable_dataset.shuffle(dataset_size)
    batched_dataset = shuffled_dataset.batch(batch_size)
    return {"dataset": batched_dataset, "dataset_resize_op": dataset_resize_op}


def prepare_training(batch_size, learning_rate, increment_dataset):
    """Create the nn, cost, iterator and optimizer for the training...

    Args:
        batch_size: Size of the batch of the training dataset.
        learning_rate: Initial learning rate for the AdamOptimizer
        increment_dataset: The number of registers to increment each training. 
            Ex: First train: 10 registers. Next train: 20 registers. increment_dataset = 10

    Returns:
        A dictionary with the following objects:
            cost: The cost function for training.
            optimizer: The optimizer functions for training.
            dataset_resize_op: The resize operation to increment the number of registers to take each training.
            iterator_initializer: Each time this operation is called, the iterator initialize 
                again with the new registers (if dataset_resize_operation is called, otherwise 
                will initialize with the same number of registers of last time)
            summaries: A list with all the tf.summary.scalar of all layers of the nn.
            saved_variables: A dictionary with all the variables to be saved.

    """
    dataset = create_training_dataset(batch_size, increment_dataset)
    iterator = create_iterator(dataset["dataset"])
    prediction, summaries, saved_variables = create_model(
        iterator["feature"], False)
    cost = tf.losses.mean_squared_error(
        labels=iterator["label"], predictions=prediction)  # Mean Squared Error
    summaries.append(tf.summary.scalar("Cost", cost))
    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate).minimize(cost)
    return {"cost": cost, "optimizer": optimizer, "dataset_resize_op": dataset["dataset_resize_op"], "iterator_initializer": iterator["initializer"], "summaries": summaries, "saved_variables": saved_variables}


def train(sess, batch_size, train_number, iterator_initializer, optimizer, cost, file_writer, summaries):
    """Train a neural network...

    Trains the neural network in sess. Executes the optimizer and cost operations until the iterator is empty.

    Args:
        batch_size: Size of the batch of the training dataset.
        train_number: The number of the current training. It is going to be used for summaries, in the graphics generated
            in tensorboard, the step number is this number, adding 1 each iteration of the training.
        iterator_initializer: It is used to initialize the iterator. So once it is initialized, the training begins.
        optimizer: This operation is used to adjust the neural networks weights, runs each iteration.
        cost: An operation that calcules the losses number on each iteration, comparing the label and the prediction. It is used
            to obtain the avg_cost of the training.
        file_writer: It has to be an instance of tf.summary.fileWriter(). It will save the summaries.
        summaries: This summaries are all the variables that will be saved with the file_writer. Has to be instance of tf.summary.merge

    Returns:
        avg_cost: The average cost of the training.

    """
    sess.run(iterator_initializer)
    avg_cost = 0
    iterations = 0
    while True:
        try:
            _, cost_aux = sess.run([optimizer, cost])
            avg_cost += cost_aux
            iterations += 1
            if iterations % 2 == 0:
                summary_values = sess.run(summaries)
                file_writer.add_summary(
                    summary_values, train_number*10+iterations)
        except tf.errors.OutOfRangeError:
            avg_cost /= (iterations*batch_size)
            break
    return avg_cost
