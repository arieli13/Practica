from Dataset import Dataset, ABCMeta, abstractmethod

class IncrementalDataset(Dataset):

    __metaclass__ = ABCMeta

    def __init__(self, dataset_path, increment_size, input_number, output_number, separator, lines_ignore, take=None):
        super(IncrementalDataset, self).__init__(dataset_path, input_number, output_number, separator, lines_ignore, take)
        self._increment_size = increment_size
        self._size_full_dataset = 0
        self._total_increments = 0
        self._increments_done = 0
        self._full_dataset = None
    
    @abstractmethod
    def increment_dataset(self): pass

    def dataset_out_of_range(self):
        return self._actual_index >= self._size
    
    def increment_out_of_range(self):
        return self._increments_done >= self._num_of_increments
