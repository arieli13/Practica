from Log import Log

class LogString(Log):

    def __init__(self, path, file_mode, header=None, separator=",", iteration=0 ):
        super(LogString, self).__init__(path, file_mode, header, separator, iteration)
    
    def log_string(self, variables):  # variables is a list
        self._data_buffer.append(str(self._iteration)+self._separator+self._separator.join(map(str, variables))+"\n")
        self._iteration += 1