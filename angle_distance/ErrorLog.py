import tensorflow as tf


class ErrorLog:

    def __init__(self, path, file_mode, iteration=0, separator=","):
        self.__file = open(path, file_mode)
        self.__iteration = iteration
        self.__errors = ["iteration%strain_error%stest_error\n"
                         % (separator, separator)]
        self.__separator = separator

    def log_error(self, training_error, testing_error):
        self.__errors.append("%d%s%f%s%f\n" % (self.__iteration,
                             self.__separator, training_error,
                             self.__separator, testing_error))
        self.__iteration += 1
    
    def save_errors(self):
        self.__file.write("".join(self.__errors))
        self.__errors = []
    
    def close_file(self):
        if len(self.__errors) > 0:
            self.save_errors()
        self.__file.close() 