import tensorflow as tf


def parse_line(line):
    distance_t, angle_t, angle_act, distance_t1, angle_t1 = tf.decode_csv(
        line, [[0.0], [0.0], [0.0], [0.0], [0.0]], field_delim=" ")
    features = tf.stack([distance_t, angle_t, angle_act])
    labels = tf.stack([distance_t1, angle_t1])
    return features, labels


def create_dataset(path):
    dataset = tf.data.TextLineDataset(path)
    dataset = dataset.map(parse_line)
    return dataset


def create_iterator(dataset):
    iterator = dataset.make_initializable_iterator()
    initializer = iterator.make_initializer(dataset)
    feature, label = iterator.get_next()
    return {"iterator": iterator, "initializer": initializer, "feature": feature, "label": label}
