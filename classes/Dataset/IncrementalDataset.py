"""IncrementalDataset class.

Load a dataset from a csv and iterates over it 
index by index. Start with N registers and than a method can be called
to increment the registers.
"""
from Dataset import Dataset, ABCMeta, abstractmethod


class IncrementalDataset(Dataset):
    """Load a dataset from a csv and iterates over it index by index.

    The dataset start with the number of registers defined by the user
    in the constructor (increment_size), and each time the
    increment_dataset() method is called, it adds the number of registers
    defined in the same attribute.
    """

    __metaclass__ = ABCMeta

    def __init__(self, dataset_path, increment_size, input_number,
                 output_number, separator, lines_ignore, 
                 buffer_size, take=None):
        """Initialize the atributes.

        Args:
            dataset_path: Path of the csv file.
            increment_size: Number of the beginning number of registers
                            that start on the dataset. Also is the number
                            of registers to increment.
            input_number: Number of input nodes of the NN.
            output_number: Number of output nodes of the NN.
            separator: Separator between columns on the csv file.
            lines_ignore: Number of lines to ignore at the beginning of the
                          csv.
            buffer_size: Size of the buffer of the dataset.
            take: Number of lines to read from the csv.
        """
        super(IncrementalDataset, self).__init__(dataset_path, input_number,
                                                 output_number, separator,
                                                 lines_ignore, take)
        self._increment_size = increment_size
        self._size_full_dataset = 0
        self._total_increments = 0
        self._increments_done = 0
        self._full_dataset = None
        self._num_of_increments = 0
        self._buffer_size = buffer_size

    @abstractmethod
    def increment_dataset(self):
        """Increment the number of registers of the dataset."""
        pass

    def dataset_out_of_range(self):
        """Return True if the dataset has been iterated at all."""
        return self._actual_index >= self._size

    def increment_out_of_range(self):
        """Return True if the are no more registers to increment."""
        return self._increments_done >= self._num_of_increments
