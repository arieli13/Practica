import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.tools import inspect_checkpoint as chkp
import re
from LogClass import LogClass

cli_str = ">> "

load_dir = ""
save_dir = "./default/"
model = []
selected_layer = []
weight = []
global_variables = []
last_commands = []

def create_new_variables(file_name, tensor_name, all_tensors, all_tensor_names=False):
    try:
        dictionary = {}
        variables_names = []
        variables = []
        reader = pywrap_tensorflow.NewCheckpointReader(file_name)
        var_to_shape_map = reader.get_variable_to_shape_map()
        for key in sorted(var_to_shape_map):
            key_aux = key.split("/")
            var = (key_aux[-1].split(":")[0],  "".join(key_aux[:len(key_aux)-1]))
            variables_names.append(var)

            with tf.name_scope(var[1]):
                variables.append(tf.Variable(reader.get_tensor(key), name=var[0]))
                
        return variables
    except Exception as e:
        print e.message
            
            
def help_function():
    string = "model -> Shows the model info."

def model(command):
    print "MODEL"

def layer(command):
    global selected_layer, global_variables
    if len(global_variables) == 0:
        print "Variables not loaded"
    else:
        pass

def weight(command):
    print "WEIGHT"

def evaluate(command):
    print "EVAL"

def save(command):
    print "SAVE"

def load_aux(dir, sess):
    try:
        global global_variables, load_dir
        global_variables = create_new_variables(tf.train.latest_checkpoint(dir), tensor_name='', all_tensors=False, all_tensor_names=True)
        sess.run(tf.global_variables_initializer())
        load_dir = dir
        for i in global_variables:
            print i.name
        print "Variables successfully loaded"
    except Exception as e:
        print e.message

def load(command, sess):
    if len(command) == 1:
        global load_dir
        if load_dir == "":
            print "Could not load. No path available."
        else:
            load_aux(load_dir, sess)
    else:
        load_aux(command[1], sess)

def command_line(sess):
    command = raw_input(cli_str)
    last_commands.append(command)

    if re.match( "model\s*$", command):
        model(re.split("\s", command))
    elif re.match( "layer\s*$|layer\s[\w]+$", command):
        layer(re.split("\s", command))
    elif re.match("weight\s[\w]+$|weight\s[\w]+\s[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$", command):
        weight(re.split("\s", command))
    elif re.match( "eval\s+csv\s+.+\s+\d+\s+\d+$", command):
        evaluate(re.split("\s", command))
    elif re.match( "save\s*$|save\s[^\s]+$", command):
        save(re.split("\s", command))
    elif re.match( "load\s*$|load\s[^\s]+$", command):
        load(re.split("\s", command), sess)
    else:
        print "Could not recognize command"

def main():
    with tf.Session() as sess:
        while True:
            command_line(sess)

if __name__ == '__main__':
    main()