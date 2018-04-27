import tensorflow as tf
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.tools import inspect_checkpoint as chkp
import re

class LogClass:

    
    def __init__(self, sess, test_function):
        self._sess = sess
        self._variable = None
        self._test_function = test_function
        self._logs = []
        self._save_dir = None
        self._save_variables = []
        self._cli_str = ">> "
        self._save_ckpt = 0
    
    
    def help(self):
        string = "\tCommand\t\t\t\t\tDescription\n"
        string += "\tvs\t\t\t\t\tShow all variables available\n"
        string += "\tv select [variable_number]\t\tSelect a variable\n"
        string += "\tv\t\t\t\t\tShow the selected variable\n"
        string += "\tv [number,]\t\t\t\tShow the index of the selected variable\n"
        string += "\tv [number,]+ (operation) number\t\tSelect the indices and assign new value to a weigth\n\t\t\t\t\t\tof the selected variable. Operation can be: =, +=, -=, *=, /=\n"
        string += "\tv undo\t\t\t\t\tUndo a previous operation made on the selected variable\n"
        string += "\tv restore\t\t\t\tRestore the selected variable to its original state\n"
        string += "\tm undo\t\t\t\t\tUndo a previous operation made on the variables\n"
        string += "\tt\t\t\t\t\tExecutes the function of test\n"
        string += "\ts path saved_variables\t\t\tAssign the path for saving and the number of the variables to be saved\n"
        string += "\ts\t\t\t\t\tSave the selected variables on the specified path\n"
        string += "\texit\t\t\t\t\tGet out of the program\n"
        print string

    
    def model_undo(self):
        try:
            variables = {}
            tf_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            for i in range(len(tf_variables)):
                variables[tf_variables[i].name] = tf_variables[i]

            if len(self._logs) > 0:
                variable = self._logs.pop()
                self._sess.run(tf.assign(variables[variable["name"]], variable["data"]))
            else:
                print "The model is in its original state"
        except Exception as e:
            print "An error raised: %s"%(e.message)
    

    def model_restore(self):
        try:
            variables = {}
            tf_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            for i in range(len(tf_variables)):
                variables[tf_variables[i].name] = tf_variables[i]

            if len(self._logs) > 0:
                for log in self._logs:
                    print log["name"]
                    self._sess.run(tf.assign(variables[log["name"]], log["data"]))
                self._logs = []
            else:
                print "The model is in its original state"
        except Exception as e:
            print "An error raised: %s"%(e.message)


    def create_log(self):
        log = {}
        log["name"] = self._variable.name
        log["data"] = self._sess.run(self._variable)
        return log

    
    def get_command(self, command):
        command = command.split(" ")
        command = [x for x in command if x]
        return command

    
    def variable_select(self, command):
        try:
            command = self.get_command(command)
            self._variable = None
            self._variable = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)[int(command[2])]
        except Exception as e:
            print "An error raised: %s"%(e.message)
    
    
    def variable_view(self, command):
        try:
            if self._variable is None:
                print "No variable selected"
            else:
                command = self.get_command(command)
                if len(command) == 1:
                    print self._sess.run(self._variable)
                    print self._variable
                else:
                    indices = command[1]
                    indices = [int(x) for x in indices.split(",") if x]
                    view_tensor = self._variable[indices[0]]
                    for index in indices[1:]:
                        view_tensor = view_tensor[index]
                    print self._sess.run(view_tensor)
        except Exception as e:
            print "An error raised: %s"%(e.message)
    
    
    def variable_restore(self):
        if self._variable is None:
            print "No variable selected"
        else:
            try:
                index = 0
                while index < len(self._logs):
                    if(self._logs[index]["name"] == self._variable.name):
                        self._sess.run(tf.assign(self._variable, self._logs[index]["data"]))
                        del self._logs[index]
                        index = 0
                print "Variable restored"
            except Exception as e:
                print "An error raised: %s"%(e.message)
    
    
    def variable_undo(self):
        if self._variable is None:
            print "No variable selected"
        else:
            try:
                tamannio_logs = len(self._logs)
                while tamannio_logs > 0:
                    if(self._logs[tamannio_logs-1]["name"] == self._variable.name):
                        self._sess.run(tf.assign(self._variable, self._logs[tamannio_logs-1]["data"]))
                        del self._logs[tamannio_logs-1]
                        return
                    tamannio_logs -= 1
                print "Variable is in its original state"
            except Exception as e:
                print "An error raised: %s"%(e.message)
    
    
    def variable_assign(self, command):
        if self._variable is None:
            print "No variable selected"
        else:
            command = self.get_command(command)
            try:
                log = self.create_log()
                indices = [[int(x) for x in command[1].split(",") if x]]

                if command[2] == "=":
                    constant_one = tf.constant(1.0, shape=self._variable.shape)
                    value_neg_one = -1.0
                    delta = tf.SparseTensor(indices, [value_neg_one], self._variable.shape)
                    result = constant_one + tf.sparse_tensor_to_dense(delta)
                    result = self._variable * result
                    new_value = float(command[3])
                    delta = tf.SparseTensor(indices, [new_value], self._variable.shape)
                    result = result + tf.sparse_tensor_to_dense(delta)
                elif command[2] == "+=":
                    constant_zero = tf.constant(0.0, shape=self._variable.shape)
                    new_value = float(command[3])
                    delta = tf.SparseTensor(indices, [new_value], self._variable.shape)
                    result = constant_zero + tf.sparse_tensor_to_dense(delta)
                    result = self._variable + result
                elif command[2] == "-=":
                    constant_zero = tf.constant(0.0, shape=self._variable.shape)
                    new_value = float(command[3])
                    delta = tf.SparseTensor(indices, [new_value], self._variable.shape)
                    result = constant_zero + tf.sparse_tensor_to_dense(delta)
                    result = self._variable - result
                elif command[2] == "*=":
                    constant_one = tf.constant(1.0, shape=self._variable.shape)
                    new_value = float(command[3])-1.0
                    delta = tf.SparseTensor(indices, [new_value], self._variable.shape)
                    result = constant_one + tf.sparse_tensor_to_dense(delta)
                    result = self._variable * result
                elif command[2] == "/=":
                    constant_one = tf.constant(1.0, shape=self._variable.shape)
                    new_value = float(command[3])-1.0
                    delta = tf.SparseTensor(indices, [new_value], self._variable.shape)
                    result = constant_one + tf.sparse_tensor_to_dense(delta)
                    result = self._variable / result
                self._sess.run(tf.assign(self._variable, result))
                self._logs.append(log)

            except Exception as e:
                print "Error: %s" % (e.message)

    
    def variables(self):
        print "Avaliable variables:"
        number = 0
        for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            print "%d.\t%s"%(number, i)
            number += 1


    def test(self):
        if self._test_function is not list:
            self._test_function()

    
    def save(self):
        try:
            if self._save_dir is None:
                print "No save direction available"
            elif len(self._save_variables) == 0:
                print "No save variables available"
            else:
                tf_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
                saved_variables = {}
                for index in self._save_variables:
                    variable = tf_variables[index]
                    saved_variables[variable.name] = variable
                saver = tf.train.Saver(saved_variables)
                saver.save(self._sess, self._save_dir, global_step=self._save_ckpt)
                self._save_ckpt += 1
        except Exception as e:
            print "An error raised: %s"%(e.message)

    
    def save_config(self, command):
        command = self.get_command(command)
        try:
            self._save_dir = command[1]
            self._save_variables = [int(x) for x in command[2].split(",") if x]
        except Exception as e:
            print "An error raised: %s"%(e.message) 

    
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

        if re.match("m\s+undo\s*$", command):
            self.model_undo()
        elif re.match("m\s+restore\s*$", command):
            self.model_restore()
        elif re.match("vs\s*$", command):
            self.variables()
        elif re.match("v", command):
            if re.match("v\s*$", command):
                self.variable_view(command)
            elif re.match("v\s+(\d,|\d)+\s*$", command):
                self.variable_view(command)
            elif re.match("v\s+(\d,|\d)+\s+(=|\+=|-=|\*=|/=)\s+[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?(%)?\s*$", command):
                self.variable_assign(command)
            elif re.match("v\s+select\s+\d+\s*$", command):
                self.variable_select(command)
            elif re.match("v\s+undo\s*$", command):
                self.variable_undo()
            elif re.match("v\s+restore\s*$", command):
                self.variable_undo()
            else:
                print "Could not recognize command"
        elif re.match("t\s*$", command):
            self.test()
        elif re.match("s\s+\S+\s+(\d,|\d)+$", command):
            self.save_config(command)
        elif re.match("s\s*$", command):
            self.save()
        elif re.match("h\s*$", command):
            self.help()
        elif re.match("exit\s*$", command):
            return
        else:
            print "Could not recognize command. Type h for help"
        self.command_line()
        