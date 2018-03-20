#!/usr/bin/env
import tensorflow as tf
from model import create_model
import math


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

def load_data(path):
    data = []
    with open(path) as f:
        f = f.readlines()
        for line in f:
            line = line.split(" ")
            line = [float(i) for i in line]
            features = [line[:8]]
            label = [[line[8]]]
            data.append([features, label])
        return data

def execute_prediction():
    dataset = load_data("./datasets/normalizado/pnode01_03000.txt")
    features = tf.placeholder(tf.float32, [None, 8])
    label = tf.placeholder(tf.float32, [None, 1])
    prediction, _, variables = create_model(features, False)
    _, cost = tf.metrics.root_mean_squared_error(
        labels=label, predictions=prediction)  # Mean Squared Error
    

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer()) 
        saver = tf.train.Saver(variables)
        _ = load_model(sess, saver, "./checkpoints/")
        dataset_len = len(dataset)
        avg_cost = 0
        buffer = []
        errors = []
        for data in dataset:
            f = data[0]
            l = data[1]
            prediction_aux, cost_aux = sess.run([prediction, cost], feed_dict={features:f, label:l})
            avg_cost += cost_aux
            buffer_string = "%f;%f\n"%(prediction_aux[0][0], l[0][0])
            print buffer_string
            buffer.append(buffer_string)
        buffer = "".join(buffer)
        with open("predictions.csv", "w") as f:
            f.write(buffer)
        final_cost = avg_cost/dataset_len
        print "Cost: %f"%(final_cost)




def main():
    execute_prediction()


if __name__ == '__main__':
    main()
