from abc import ABCMeta, abstractmethod

class Dataset:

    __metaclass__ = ABCMeta
    
    def __init__(self, dataset_path, input_number, output_number, separator, lines_ignore, take=None):
        self._dataset_path = dataset_path
        self._input_number = input_number
        self._output_number = output_number
        self._separator = separator
        self._lines_ignore = lines_ignore

        self._actual_index = 0  # To iterate over the dataset. The function get_next uses it.
        self._size = 0
        self._dataset = None
        self._take = take

    @abstractmethod
    def get_next(self): pass

    
    @abstractmethod
    def _load_dataset(self): pass

    
    @abstractmethod
    def shuffle(self): pass

   
    @abstractmethod
    def dataset_out_of_range(self): pass

    def restore_index(self): 
        self._actual_index = 0

    
    def get_size(self):
        return self._size