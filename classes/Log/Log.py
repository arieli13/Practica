from abc import ABCMeta, abstractmethod

class Log:

    __metaclass__ = ABCMeta

    def __init__(self, path, file_mode, header=None, separator=",", iteration=0):
        self.__file = open(path, file_mode)
        self._data_buffer = []
        if header is not None:
            self._data_buffer.append(header)
        self._separator = separator
        self._iteration = iteration
    
    def save(self):
        self.__file.write("".join(self._data_buffer))
        self._data_buffer = []


    def close_file(self): 
        if len(self._data_buffer) > 0:
            self.save()
        self.__file.close()