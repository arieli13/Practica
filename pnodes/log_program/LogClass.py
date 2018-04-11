import tensorflow as tf
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.tools import inspect_checkpoint as chkp
import re

class LogClass:

    def __init__(self, sess, prediction):
        self._sess = sess
        self._variable = None
        self._prediction = prediction

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
                indices = [[int(x) for x in command[0].split(",") if x]]
                weight = tf.gather(self._variable, indices[0])
                for index in indices:
                    weight = tf.gather(weight, index)
                self._sess.run(tf.assign(weight, float(command[2])))
        else:
            try:
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

    def evaluate(self, command):
        print "EVAL"

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
        elif re.match("variable\s*$|variable\s+[\d+]+$|variable\s+[\d,]+[\d,*]\s+=\s+[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$", command):
            self.variable([x for x in command.split(" ")[1:] if x])
        elif re.match("weight\s[\w]+$|weight\s[\w]+\s[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$", command):
            self.weight([x for x in command.split(" ")[1:] if x])
        elif re.match("variables\s*$", command):
            self.variables()
        elif re.match("eval\s+csv\s+.+\s+\d+\s+\d+$", command):
            self.evaluate([x for x in command.split(" ")[1:] if x])
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