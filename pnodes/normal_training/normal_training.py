import tensorflow as tf
import math
import random

dataset_train_path = "../datasets/normalizado/pnode03_03000_train.txt"
dataset_test_path = "../datasets/normalizado/pnode03_03000_test.txt"
dataset_full_path = "../datasets/normalizado/pnode03_03000.txt"
##################
n_inputs = 8
n_outputs = 1
hidden_layers_nodes = [20, 20]
dropout_rate = [0.1, 0.1]

learning_rate = 0.001

batch_size = 10
iterations = 200
##################
training = tf.Variable(True)
negate_training = tf.assign(training, tf.logical_not(training))

X = tf.placeholder(tf.float32, [None, n_inputs])
Y = tf.placeholder(tf.float32, [None, n_outputs])

def load_dataset(path):
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

def create_model():

    saved_variables = {}

    w1 = tf.Variable(tf.random_normal([n_inputs, hidden_layers_nodes[0]], 0.0, 0.1, tf.float32))
    b1 = tf.Variable(tf.random_normal([hidden_layers_nodes[0]], 0.0, 0.1, tf.float32))
    o1 = tf.add(tf.matmul(X, w1), b1)
    a1 = tf.nn.relu(o1)
    l1 = tf.layers.dropout(a1, dropout_rate[0], training=training) 

    w2 = tf.Variable(tf.random_normal([hidden_layers_nodes[0], hidden_layers_nodes[1]], 0.0, 0.1, tf.float32))
    b2 = tf.Variable(tf.random_normal([hidden_layers_nodes[1]], 0.0, 0.1, tf.float32))
    o2 = tf.add(tf.matmul(l1, w2), b2)
    a2 = tf.nn.relu(o2)
    l2 = tf.layers.dropout(a2, dropout_rate[1], training=training) 

    wo = tf.Variable(tf.random_normal([hidden_layers_nodes[1], n_outputs], 0.0, 0.1, tf.float32))
    bo = tf.Variable(tf.random_normal([n_outputs], 0.0, 0.1, tf.float32))
    output_layer = tf.nn.sigmoid(tf.add(tf.matmul(l2, wo), bo))

    saved_variables[w1.name] = w1
    saved_variables[w2.name] = w2
    saved_variables[wo.name] = wo
    saved_variables[b1.name] = b1
    saved_variables[b2.name] = b2
    saved_variables[bo.name] = bo

    return output_layer, saved_variables

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

prediction, saved_variables = create_model()
cost_mse = tf.losses.mean_squared_error(labels=Y, predictions=prediction)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost_mse)

_, cost_rmse = tf.metrics.root_mean_squared_error(Y, prediction)

def train(sess, saver, ckpt):
    training_dataset = load_dataset(dataset_train_path)
    testing_dataset = load_dataset(dataset_test_path)

    training_dataset_size = len(training_dataset)
    testing_dataset_size = len(testing_dataset)
    
    assert training_dataset_size>=batch_size

    total_epochs = int(math.ceil(float(training_dataset_size)/float(batch_size)))
    
    log = open("LOG.csv", "w+")

    for iteration in range(iterations):
    #iteration = 1
    #while True:
        min_index = 0
        avg_cost_train = 0
        random.shuffle(training_dataset)
        #Begin training
        for _ in range(total_epochs):
            data = training_dataset[min_index:(min_index+batch_size)]
            x = [i[0] for i in data ]
            y = [i[1] for i in data]
            min_index+=batch_size
            _, cost_aux = sess.run([optimizer, cost_mse], feed_dict={X:x, Y:y})
            avg_cost_train+=cost_aux

        #Begin testing
        avg_cost_test = 0
        sess.run(negate_training)
        for i in range(testing_dataset_size):
            data = testing_dataset[i:i+1]
            x = [i[0] for i in data]
            y = [i[1] for i in data]
            min_index+=batch_size
            cost_aux = sess.run(cost_rmse, feed_dict={X:x, Y:y})
            avg_cost_test+=cost_aux
        sess.run(negate_training)
        print "Iteration: %d\ttrain cost: %f\ttest cost: %f"%(iteration, avg_cost_train/training_dataset_size, avg_cost_test/testing_dataset_size)
        ckpt+=1


        save_model(sess, saver, "./checkpoints/", ckpt)
        log.write( "%d;%f;%f\n"%(iteration+1, avg_cost_train/training_dataset_size, avg_cost_test/testing_dataset_size) )

        #iteration+=1


def test():
    full_dataset = load_dataset(dataset_full_path)
    full_dataset_size = len(full_dataset)
    avg_cost = 0
    
    predictions = open("predictions.csv", "w+")
    sess.run(negate_training)
    for data in full_dataset:
        predic, cost_aux = sess.run([prediction, cost_rmse], feed_dict={X:[data[0]], Y:[data[1]]})
        predictions.write("%f;%f\n"%(predic[0][0], data[1][0]))
        avg_cost+=cost_aux
    print "Full dataset avg cost rmse: %f"%(avg_cost/full_dataset_size)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    saver = tf.train.Saver(saved_variables)
    ckpt = load_model(sess, saver, "./checkpoints/")
    train(sess, saver, ckpt)
    test()

#plot "LOG.csv" using 2 with lines title "train", "" using 3 with lines title "test"
