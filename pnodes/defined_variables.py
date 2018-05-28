import tensorflow as tf
import sys
sys.path.append("../classes/Dataset")
from ListClassicDataset import ListClassicDataset
from ListIncrementalDataset import ListIncrementalDataset

n_inputs = 8  # Inputs for the NN
n_outputs = 1  # Outputs of the NN

hidden_layers_nodes = [10, 7]  # Nodes in each hidden layer
hidden_layers_ac_fun = [tf.nn.relu, tf.nn.relu]  # Activation functions of each hidden layers
dropout_rate = [0.1, 0.1, 0.1]  # Dropout rate of te hidden layers

learning_rate = 0.1

batch_size = 1  # Batch size of the TRAINING dataset

training = tf.Variable(True)  # If true than dropout is activated else it is not
mode = tf.placeholder(tf.bool)  
training_mode_op = tf.assign(training, mode)  # Operation to change the training variable value

X = tf.placeholder(tf.float32, [None, n_inputs])  # Inputs for the NN
Y = tf.placeholder(tf.float32, [None, n_outputs])  # Labels for the NN

training_finish_reading = 500
memory_size = 100
iterations = 10000  # Number of iterations of the training

training_increment_dataset = ListIncrementalDataset("./datasets/normalized/pnode0.csv",
                                                    1, n_inputs, n_outputs, ",", 1, memory_size, training_finish_reading, 20)

training_dataset = ListClassicDataset("./datasets/normalized/pnode0.csv",
                                batch_size, n_inputs, n_outputs, ",", 1, training_finish_reading)

testing_dataset = ListClassicDataset("./datasets/normalized/pnode0.csv", 1,
                               n_inputs, n_outputs, ",", training_finish_reading+1)
full_dataset = ListClassicDataset("./datasets/normalized/pnode0.csv", 1,
                            n_inputs, n_outputs, ",", 1)

weight_log_path = "/logs_500"
error_log_path = "./error_log_5.csv"
predictions_log_path = "predictions_log_500.csv"