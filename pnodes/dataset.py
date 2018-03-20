"""Creates a iterator and a dataset..."""

import tensorflow as tf


def parse_line(line):
    """Take an string input and returns the features and labels for a neural network...

    Args:
        line: A string as follows: "distance_t angle_t angle_act distance_t1 angle_t1"

    Returns:
        features: tf.stack() with a list of floats inside as follows: [distance_t, angle_t, angle_act]
        labels: tf.stack() with a list of floats inside as follows: [distance_t1, angle_t1]

    """
    ball_in_right_hand, ball_dist, box_size, ball_in_left_hand, box_ang, ball_ang, box_dist, ball_size, Confidence = tf.decode_csv( line, [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], field_delim=" ")
    features = tf.stack([ball_in_right_hand, ball_dist, box_size, ball_in_left_hand, box_ang, ball_ang, box_dist, ball_size])
    labels = tf.stack([Confidence])
    return features, labels


def create_dataset(path):
    """Create a dataset from a specific csv...

    Each line of the csv must be separated with carriage return and each value of the line must be separated with blanck space.
    Have to be 5 float values each line.

    Args:
        path: The path of the csv

    Returns:
        dataset: The dataset of the csv passed over parameter.

    """
    dataset = tf.data.TextLineDataset(path)
    dataset = dataset.map(parse_line)
    return dataset


def create_iterator(dataset):
    """Create a iterator from a dataset...

    Args:
        dataset: A previous created dataset. Has to be an instance of tf.data.Dataset.

    Return:
        Returns a dictionary with following values:
            iterator: The initialized iterator for the dataset.
            initializer: The initializer of the iterator. To restart it each time it is empty.
            feature: The feature returned from the iterator to train the neural network.
            label: The label returned from the iterator to compare with prediction of the nn.  

    """
    iterator = dataset.make_initializable_iterator()
    initializer = iterator.make_initializer(dataset)
    feature, label = iterator.get_next()
    return {"iterator": iterator, "initializer": initializer, "feature": feature, "label": label}
