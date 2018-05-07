import tensorflow as tf

class WeightLog:
    #"iteration%slayer%srow_col%svalue\n"
    def __init__(self, variables, sess, log_path, header, separator=","):
        self.__variables = variables
        self.__num_variables = len(self.__variables)
        self.__data_buffer = [header]
        self.__sess = sess
        self.__file = open(log_path, "w+")
        self.__log_number = 0
        self.__variables_size = []  # Stores the size of the variables 
        self.__separator = separator
        self.__calculate_variables_size()

    
    def log_weights(self):
        variables = self.__sess.run(self.__variables)
        for v in range(self.__num_variables):
            len_v = len(variables[v])
            for r in range(len_v):
                len_r = len( variables[v][r] )
                for c in range(len_r):
                    self.__data_buffer.append(str(self.__log_number)+self.__separator+str(v)+self.__separator+str(r)+"_"+str(c)+self.__separator+str(variables[v][r][c])+"\n" )
        self.__log_number += 1
    

    def __calculate_variables_size(self):
        variables = self.__sess.run(self.__variables)
        for v in range(self.__num_variables):
            len_r = len(variables[v])  # Rows
            len_c = len( variables[v][0] )  # Cols
            self.__variables_size.append( (len_r, len_c) )

    def save_log(self):
        self.__file.write("".join(self.__data_buffer))
        self.__data_buffer = []
    
    def close_file(self):
        if len(self.__data_buffer) > 0:
            self.save_log()
        self.__file.close()
                

        


