from Log import Log

class LogWeight(Log):

    def __init__(self, path, file_mode, variables, sess, header=None, 
                 separator=",", iteration=0):
        super(LogWeight, self).__init__(path, file_mode, header, separator, iteration)
        self.__variables = variables
        self.__num_variables = len(self.__variables)
        self.__sess = sess
        self.__variables_size = []  # Stores the size of the variables 
        self.__calculate_variables_size()
    
    def log_weights(self):
        variables = self.__sess.run(self.__variables)
        for v in range(self.__num_variables):
            len_v = len(variables[v])
            for r in range(len_v):
                len_r = len( variables[v][r] )
                for c in range(len_r):
                    self._data_buffer.append(str(self._iteration)+self._separator+str(v)+self._separator+str(r)+"_"+str(c)+self._separator+str(variables[v][r][c])+"\n" )
        self._iteration += 1
    

    def __calculate_variables_size(self):
        variables = self.__sess.run(self.__variables)
        for v in range(self.__num_variables):
            len_r = len(variables[v])  # Rows
            len_c = len( variables[v][0] )  # Cols
            self.__variables_size.append( (len_r, len_c) )