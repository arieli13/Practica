"""Creates the dataset for the testing, cost and tests a neural network..."""

import tensorflow as tf
from dataset import create_dataset, create_iterator
from model import create_model


def create_testing_dataset():
    """Create the dataset of testing...

    Returns:
        A dictionary with the variables:
            dataset: The dataset for training.
            dataset_resize_op: The resize operation to increment the number of registers to take each training.

    """
    complete_dataset = create_dataset(
        "./datasets/normalizado/pnode01_03000_test.txt")
    batched_dataset = complete_dataset.batch(1)
    return batched_dataset


def prepare_testing():
    """Create the nn, cost_update for the training...

    Returns:
        A dictionary with the following objects:
            cost_update: Operation to update the total count of values to calculate the SMSE. Returns the cost.
            iterator_initializer: Each time this operation is called, the iterator initialize 
                again with the new registers (if dataset_resize_operation is called, otherwise 
                will initialize with the same number of registers of last time)
            summaries: A list with all the tf.summary.scalar of all layers of the nn.

    """
    dataset = create_testing_dataset()
    iterator = create_iterator(dataset)
    prediction, _, _ = create_model(iterator["feature"], False)
    cost, update = tf.metrics.root_mean_squared_error(
        labels=iterator["label"], predictions=prediction)  # Mean Squared Error
    summaries = [tf.summary.scalar("cost_test", cost)]
    return {"cost_update": update, "iterator_initializer": iterator["initializer"], "summaries": summaries}


def test(sess, train_number, iterator_initializer, cost_update, file_writer, summaries):
    """Test a neural network...

    Tests the neural network in sess. Executes the cost operation until the iterator is empty.

    Args:
        train_number: The number of the current training. It is going to be used for summaries, in the graphics generated
            in tensorboard, the step number is this number, adding 1 each iteration of the training.
        iterator_initializer: It is used to initialize the iterator. So once it is initialized, the training begins.
        cost: An operation that calcules the losses number on each iteration, comparing the label and the prediction. It is used
            to obtain the avg_cost of the training.
        file_writer: It has to be an instance of tf.summary.fileWriter(). It will save the summaries.
        summaries: This summaries are all the variables that will be saved with the file_writer. Has to be instance of tf.summary.merge

    Returns:
        avg_cost: The average cost of the test.

    """
    sess.run(iterator_initializer)
    avg_cost = 0
    iterations = 0
    while True:
        try:
            cost_aux = sess.run(cost_update)
            avg_cost += cost_aux
            iterations += 1
            if iterations % 10 == 0:
                summary_values = sess.run(summaries)
                file_writer.add_summary(
                    summary_values, train_number*10+iterations)
        except tf.errors.OutOfRangeError:
            avg_cost /= iterations
            break
    return avg_cost
