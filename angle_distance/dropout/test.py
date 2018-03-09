import tensorflow as tf
from dataset import create_dataset, create_iterator
from model import create_model


def create_testing_dataset():
    complete_dataset = create_dataset(
        "/home/ariel/Dropbox/UDC/angle_distance/dataset/data_last_500.txt")
    batched_dataset = complete_dataset.batch(1)
    return batched_dataset


def prepare_testing():
    dataset = create_testing_dataset()
    iterator = create_iterator(dataset)
    prediction, _, _ = create_model(iterator["feature"], False)
    cost = tf.losses.mean_squared_error(
        labels=iterator["label"], predictions=prediction)  # Mean Squared Error
    summaries = [tf.summary.scalar("cost_test", cost)]
    return {"cost": cost, "iterator_initializer": iterator["initializer"], "summaries": summaries}


def test(sess, train_number, iterator_initializer, cost, fileWriter, summaries):
    sess.run(iterator_initializer)
    avg_cost = 0
    iterations = 0
    while True:
        try:
            cost_aux = sess.run(cost)
            avg_cost += cost_aux
            iterations += 1
            if iterations % 10 == 0:
                summary_values = sess.run(summaries)
                fileWriter.add_summary(
                    summary_values, train_number*10+iterations)
        except tf.errors.OutOfRangeError:
            avg_cost /= iterations
            break
    return avg_cost
