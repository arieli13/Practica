 #!/usr/bin/env
import tensorflow as tf
from train import prepare_training, train
from test import prepare_testing, test

learning_rate = 0.0001
batch_size = 10
increment_dataset = 10  # Number of registers to add to the next training


def load_model(sess, saver, path):
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


def save_model(sess, saver, path, ckpt):
    saver.save(sess, path, global_step=ckpt)  # Saves the weights of the model


def incremental_training(sess, saver, training_variables, testing_variables, ckpt):
    file_writer_train = tf.summary.FileWriter("./logs/train/", sess.graph)
    file_writer_test = tf.summary.FileWriter("./logs/test/", sess.graph)
    summaries_testing = tf.summary.merge(testing_variables["summaries"])
    summaries_training = tf.summary.merge(training_variables["summaries"])
    file = open("./LOG.csv", "a+")
    data_buffer = []
    for train_number in range(1, 200):
        sess.run(training_variables["dataset_resize_op"])
        avg_cost_train = train(sess, batch_size, train_number,
                               training_variables["iterator_initializer"], training_variables["optimizer"], training_variables["cost"],  file_writer_train, summaries_training)

        avg_cost_test = test(
            sess, train_number, testing_variables["iterator_initializer"], testing_variables["cost"], file_writer_test, summaries_testing)

        data_log = "%d;%f;%f\n" % (
            train_number, avg_cost_train, avg_cost_test)
        data_buffer.append(data_log)
        print data_log
        if train_number % 10 == 0:
                # Writes the data_buffer on log.csv file
            file.write("".join(data_buffer))
            data_buffer = []
            save_model(sess, saver, "./checkpoints/model.ckpt", ckpt)
            ckpt += 1

    if len(data_buffer) != 0:
        file.write("".join(data_buffer))
    file.close()
    saver.save(sess, "./checkpoints/model.ckpt", global_step=ckpt)


def execute_incremental_training():
    config = tf.ConfigProto()
    config.intra_op_parallelism_threads = 2
    config.inter_op_parallelism_threads = 2

    with tf.Session(config=config) as sess:
        training_variables = prepare_training(
            batch_size, learning_rate, increment_dataset)
        testing_variables = prepare_testing()

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(training_variables["saved_variables"]) # Only saves weights and biases

        ckpt = load_model(sess, saver, "./checkpoints/")
        
        if ckpt == 0:
            save_model(sess, saver, "./checkpoints/model.ckpt", ckpt)
        incremental_training(
            sess, saver, training_variables, testing_variables, ckpt)


def main():
    execute_incremental_training()


if __name__ == '__main__':
    main()
