import tensorflow as tf
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.tools import inspect_checkpoint as chkp
import re

class LogClass:

    def __init__(self, sess, prediction, test_function):
        self._sess = sess
        self._variable = None
        self._prediction = prediction
        self._test_function = test_function

        self._cli_str = ">> "
    
    def help_function(self):
        string = "model -> Shows the model info."

    def model(self, command):
        print "MODEL"

    def variable(self, command):
        if len(command) == 0:
            if self._variable is None:
                print "No variable selected"
            else:
                try:
                    print self._sess.run(self._variable)
                    print self._variable
                except Exception as e:
                    print "An error raised: %s"%(e.message)
        
        elif len(command) == 3:
            if self._variable is None:
                print "No variable selected"
            else:
                try:

                    indices = [[int(x) for x in command[0].split(",") if x]]

                    if command[1] == "=":
                        constant_one = tf.constant(1.0, shape=self._variable.shape)
                        value_neg_one = -1.0
                        delta = tf.SparseTensor(indices, [value_neg_one], self._variable.shape)
                        result = constant_one + tf.sparse_tensor_to_dense(delta)
                        result = self._variable * result
                        new_value = float(command[2])
                        delta = tf.SparseTensor(indices, [new_value], self._variable.shape)
                        result = result + tf.sparse_tensor_to_dense(delta)
                    elif command[1] == "+=":
                        constant_zero = tf.constant(0.0, shape=self._variable.shape)
                        new_value = float(command[2])
                        delta = tf.SparseTensor(indices, [new_value], self._variable.shape)
                        result = constant_zero + tf.sparse_tensor_to_dense(delta)
                        result = self._variable + result
                    elif command[1] == "-=":
                        constant_zero = tf.constant(0.0, shape=self._variable.shape)
                        new_value = float(command[2])
                        delta = tf.SparseTensor(indices, [new_value], self._variable.shape)
                        result = constant_zero + tf.sparse_tensor_to_dense(delta)
                        result = self._variable - result
                    elif command[1] == "*=":
                        constant_one = tf.constant(1.0, shape=self._variable.shape)
                        new_value = float(command[2])-1.0
                        delta = tf.SparseTensor(indices, [new_value], self._variable.shape)
                        result = constant_one + tf.sparse_tensor_to_dense(delta)
                        result = self._variable * result
                    elif command[1] == "/=":
                        constant_one = tf.constant(1.0, shape=self._variable.shape)
                        new_value = float(command[2])-1.0
                        delta = tf.SparseTensor(indices, [new_value], self._variable.shape)
                        result = constant_one + tf.sparse_tensor_to_dense(delta)
                        result = self._variable / result
                    self._sess.run(tf.assign(self._variable, result))

                except Exception as e:
                    print "Error: %s" % (e.message)
        else:
            try:
                if command[0][-1] == ",":
                    indices = command[0][:len(command[0])-1]
                    indices = [int(x) for x in indices.split(",")]
                    view_tensor = self._variable[indices[0]]
                    for index in indices[1:]:
                        view_tensor = view_tensor[index]
                    print self._sess.run(view_tensor)
                else:
                    self._variable = None
                    self._variable = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)[int(command[0])]
            except Exception as e:
                print "An error raised: %s"%(e.message)

        
    def variables(self):
        print "Avaliable variables:"
        for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            print i

    def weight(self, command):
        print "WEIGHT"

    def test(self):
        self._test_function()

    def save(self, command):
        print "SAVE"

    def load_aux(self, dir, sess):
        """try:
            global global_variables, load_dir
            global_variables = create_new_variables(tf.train.latest_checkpoint(dir), tensor_name='', all_tensors=False, all_tensor_names=True)
            sess.run(tf.global_variables_initializer())
            load_dir = dir
            for i in global_variables:
                print i.name
            print "Variables successfully loaded"
        except Exception as e:
            print e.message
        """
        pass

    def load(self, command):
        """if len(command) == 1:
            global load_dir
            if load_dir == "":
                print "Could not load. No path available."
            else:
                load_aux(load_dir, sess)
        else:
            load_aux(command[1], sess)
        """
        for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            print i   # i.name if you want just a name

    def command_line(self):
        command = raw_input(self._cli_str)

        if re.match("model\s*$", command):
            self.model([x for x in command.split(" ")[1:] if x])
        elif re.match("variable\s*$|variable\s+(\d,)+$|variable\s+[\d,]+$|variable\s+(\d,)+\s*(=|\+=|-=|\*=|/=)\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$", command):
            self.variable([x for x in command.split(" ")[1:] if x])
        elif re.match("weight\s[\w]+$|weight\s[\w]+\s[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$", command):
            self.weight([x for x in command.split(" ")[1:] if x])
        elif re.match("variables\s*$", command):
            self.variables()
        elif re.match("test\s*$", command):
            self.test()
        elif re.match("save\s*$|save\s[^\s]+$", command):
            self.save([x for x in command.split(" ")[1:] if x])
        elif re.match("load\s*$|load\s[^\s]+$", command):
            self.load([x for x in command.split(" ")[1:] if x])
        elif re.match("exit\s*$", command):
            return
        else:
            print "Could not recognize command"
        self.command_line()
        
#python-neat