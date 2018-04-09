import tensorflow as tf
from tensorflow.python.tools import inspect_checkpoint as chkp
import re

cli_str = ">> "

load_dir = ""
save_dir = "./default/"
model = {} # {name:variable}
layer = {}  # name:variable
weight = {}  # name:variable
last_commands = []

def help_function():
    string = "model -> Shows the model info."

def model(command):
    print "MODEL"

def layer(command):
    print command

def weight(command):
    print "WEIGHT"

def evaluate(command):
    print "EVAL"

def save(command):
    print "SAVE"

def load(command, sess, saver):
    if len(command) == 1:
        if load_dir == "":
            print "Could not load. No path available."
        else:
            pass
    else:
        try:
            #saver.restore(sess, tf.train.latest_checkpoint(command[1]))
            tf.Variable(tf.random_normal(shape=[1, 1]), name="output/Variable:0")
            chkp.print_tensors_in_checkpoint_file(tf.train.latest_checkpoint(command[1]), tensor_name='HL2', all_tensors=True, all_tensor_names=True)
            
            print "Model succesfully saved"
        except Exception as e:
            print e.message

def command_line(data):
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
        load(re.split("\s", command), data["sess"], data["saver"])
    else:
        print "Could not recognize command"

def main():
    with tf.Session() as sess:
        x = tf.Variable(12)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        data = {}
        data["sess"] = sess
        data["saver"] = saver
        while True:
            command_line(data)

if __name__ == '__main__':
    main()