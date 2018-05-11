"""Class Dataset, load a csv and iterate over it."""
from abc import ABCMeta, abstractmethod


class Dataset:
    """Load a csv and treat it as a iterator."""

    __metaclass__ = ABCMeta

    def __init__(self, dataset_path, input_number, output_number, separator,
                 lines_ignore, take=None):
        """Initialize the atributes.

        Args:
            dataset_path: Path of the csv file.
            input_number: Number of input nodes of the NN.
            output_number: Number of output nodes of the NN.
            separator: Separator between columns on the csv file.
            lines_ignore: Number of lines to ignore at the beginning of the csv.
            take: Number of lines to read from the csv.
        """
        self._dataset_path = dataset_path
        self._input_number = input_number
        self._output_number = output_number
        self._separator = separator
        self._lines_ignore = lines_ignore

        self._actual_index = 0  # To iterate over the dataset.
        self._size = 0
        self._dataset = None
        self._take = take

    @abstractmethod
    def get_next(self):
        """Return registers from the dataset."""
        pass

    @abstractmethod
    def _load_dataset(self):
        """Load a csv file. Executed on constructor."""
        pass

    @abstractmethod
    def shuffle(self):
        """Shuffle random the dataset."""
        pass

    @abstractmethod
    def dataset_out_of_range(self):
        """Return True if the dataset has been iterated at all."""
        pass

    def restore_index(self):
        """Restore the index of the next register to 0. Begin iteration."""
        self._actual_index = 0

    def get_size(self):
        """Return the size of the dataset."""
        return self._size
