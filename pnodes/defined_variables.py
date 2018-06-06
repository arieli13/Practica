import tensorflow as tf
import sys
sys.path.append("../classes/Dataset")
from ListClassicDataset import ListClassicDataset
from ListIncrementalDataset import ListIncrementalDataset

n_inputs = 8  # Inputs for the NN
n_outputs = 1  # Outputs of the NN

hidden_layers_nodes = [128, 64, 64, 32]  # Nodes in each hidden layer
hidden_layers_ac_fun = [tf.nn.relu]*len(hidden_layers_nodes)  # Activation functions of each hidden layers
hidden_layers_ac_fun_names = ["relu"]*len(hidden_layers_nodes)
dropout_rate = [0.1]*len(hidden_layers_nodes)  # Dropout rate of te hidden layers
train_dropout = False
learning_rate = 0.0001

beta_1 = 0.9
beta_2 = 0.999
epsilon = 1e-8

training = tf.Variable(train_dropout)  # If true than dropout is activated else it is not
mode = tf.placeholder(tf.bool)
training_mode_op = tf.assign(training, mode)  # Operation to change the training variable value

X = tf.placeholder(tf.float32, [None, n_inputs])  # Inputs for the NN
Y = tf.placeholder(tf.float32, [None, n_outputs])  # Labels for the NN

training_finish_reading = 250
memory_size = 100
mini_batch_size = 5
train_steps = 25  # Number of train steps of the training
pnode_number = 13

def read_datasets(path, training_registers, skip):
    train = []
    validation = []
    test = []
    with open(path) as f:
        lines = f.readlines()[skip:]
        lines =  [ [[y[:8]], [y[8:]]] for y in [[float(k) for k in i.split(",")] for i in lines ]]
        train = lines[:training_registers]
        validation = lines[training_registers:]
        test = lines[:]
    return train, validation, test

train_dataset, validation_dataset, test_dataset = read_datasets("./datasets/normalized/pnode%d.csv"%(pnode_number), training_finish_reading, 1)

error_log_path = "./tests/errors/error_log"
predictions_log_path = "./tests/predictions/predictions"
time_log_path = "./tests/times/time"

checkpoints_path = "./checkpoints/"