import tensorflow as tf
import sys
sys.path.append("../classes/Dataset")
from ListClassicDataset import ListClassicDataset
from ListIncrementalDataset import ListIncrementalDataset

n_inputs = 8  # Inputs for the NN
n_outputs = 1  # Outputs of the NN

hidden_layers_nodes = [15, 10]  # Nodes in each hidden layer
hidden_layers_ac_fun = [tf.nn.relu]*len(hidden_layers_nodes)  # Activation functions of each hidden layers
dropout_rate = [0.5]*len(hidden_layers_nodes)  # Dropout rate of te hidden layers

learning_rate = 0.5

training = tf.Variable(False)  # If true than dropout is activated else it is not
mode = tf.placeholder(tf.bool)
training_mode_op = tf.assign(training, mode)  # Operation to change the training variable value

X = tf.placeholder(tf.float32, [None, n_inputs])  # Inputs for the NN
Y = tf.placeholder(tf.float32, [None, n_outputs])  # Labels for the NN

training_finish_reading = 500
memory_size = 500
train_steps = 5  # Number of train steps of the training
pnode_number = 20

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