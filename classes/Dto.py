class DTO:
    def __init__(self, separator=","):
        self.number = 0
        self.memory_size = 0
        self.train_steps = 0
        self.learning_rate = 0
        self.train_error = 0
        self.test_error = 0
        self.final_test_error = 0
        self.final_test_stddev = 0
        self.optimizer = ""
        self.nn = ""
        self.pnode = 0
        self.time = 0
        self.activation_func = ""
        self.dropout = False
        self.train_type = "" 

        self.__separator = separator
    
    def to_string(self):
        data = [self.number,self.memory_size,self.train_steps,self.learning_rate,self.train_error, self.test_error,self.final_test_error,self.final_test_stddev,self.optimizer,self.nn,self.pnode ,self.time ,self.activation_func,self.dropout,self.train_type]
        data = [str(i) for i in data]
        return self.__separator.join(data)+"\n"
