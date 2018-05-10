from abc import ABCMeta, abstractmethod

class NeuralNetwork:

    __metaclass__ = ABCMeta

    def __init__(self, sess):
        self._prediction = None
        self._optimizer = None
        self._cost_train = None
        self._cost_test = None

    @abstractmethod
    def load_datasets(self): pass


    @abstractmethod
    def load_variables(self): pass


    @abstractmethod
    def save_variables(self): pass


    @abstractmethod
    def create_model(self): pass


    @abstractmethod
    def train(self): pass


    @abstractmethod
    def test(self): pass

    