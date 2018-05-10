import tensorflow as tf
import sys
sys.path.append("../classes/Dataset")
from ListClassicDataset import ListClassicDataset

n_inputs = 4  # Inputs for the NN
n_outputs = 2  # Outputs of the NN

hidden_layers_nodes = [15]  # Nodes in each hidden layer
hidden_layers_ac_fun = [None, None, None]  # Activation functions of each hidden layers
dropout_rate = [0.1, 0.1, 0.1]  # Dropout rate of te hidden layers

learning_rate = 0.001

batch_size = 1  # Batch size of the TRAINING dataset
iterations = 110  # Number of iterations of the training

training = tf.Variable(True)  # If true than dropout is activated else it is not
mode = tf.placeholder(tf.bool)  
training_mode_op = tf.assign(training, mode)  # Operation to change the training variable value

X = tf.placeholder(tf.float32, [None, n_inputs])  # Inputs for the NN
Y = tf.placeholder(tf.float32, [None, n_outputs])  # Labels for the NN

training_dataset = ListClassicDataset("../datasets_angle_distance/cart_leftArmMovement_train.csv",
                                batch_size, n_inputs, n_outputs, ",", 1, 50)
testing_dataset = ListClassicDataset("../datasets_angle_distance/cart_leftArmMovement_test.csv", 1,
                               n_inputs, n_outputs, ",", 1)
full_dataset = ListClassicDataset("../datasets_angle_distance/cart_leftArmMovement.csv", 1,
                            n_inputs, n_outputs, ",", 1)

weight_log_path = "/logs"
error_log_path = "./error_log.csv"
predictions_log_path = "predictions_log.csv"
