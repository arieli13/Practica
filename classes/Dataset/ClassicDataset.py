"""Class ClassicDataset. Load a dataset from a csv and iterates over it index by index."""
from Dataset import Dataset, abstractmethod, ABCMeta


class ClassicDataset(Dataset):
    """Load a dataset from a csv and iterates over it index by index."""

    __metaclass__ = ABCMeta

    def __init__(self, dataset_path, batch_size, input_number, output_number,
                 separator, lines_ignore, take=None):
        """Initialize the atributes.

        Args:
            dataset_path: Path of the csv file.
            batch_size: Size of the batches to divide the dataset.
            input_number: Number of input nodes of the NN.
            output_number: Number of output nodes of the NN.
            separator: Separator between columns on the csv file.
            lines_ignore: Number of lines to ignore at the beginning of the 
                          csv.
            take: Number of lines to read from the csv.
        """
        super(ClassicDataset, self).__init__(dataset_path, input_number,
                                             output_number, separator,
                                             lines_ignore, take)
        self._batch_size = batch_size
        self._total_batches = 0

    def get_total_batches(self):
        """Return the number of batches in which the dataset has been divided."""
        return self._total_batches

    def dataset_out_of_range(self):
        """Return True if the dataset has been iterated at all."""
        return self._actual_index >= self._total_batches
