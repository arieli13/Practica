from Dataset import Dataset, abstractmethod, ABCMeta

class ClassicDataset(Dataset):

    __metaclass__ = ABCMeta

    def __init__(self, dataset_path, batch_size, input_number, output_number, separator, lines_ignore, take=None):
        super(ClassicDataset, self).__init__(dataset_path, input_number, output_number, separator, lines_ignore, take)
        self._batch_size = batch_size
        self._total_batches = 0
    
    
    def get_total_batches(self):
        return self._total_batches
    
    
    def dataset_out_of_range(self):
        return self._actual_index >= self._total_batches
    