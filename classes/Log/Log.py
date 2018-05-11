"""Log class. Stores data on a file."""
from abc import ABCMeta, abstractmethod


class Log:
    """Stores data on a file."""

    __metaclass__ = ABCMeta

    def __init__(self, path, file_mode, header=None, separator=",", iteration=0):
        """Initialize all variables.

        Args:
            path: Path of the new/old file.
            file_mode: The mode in which the file will be open.
                       (w, w+, a)
            header: First line of the file.
            separator: When working with csv files, it is the
                       separator of the cols.
            iteration: It is a counter that tells how much data has
                       been saved.

        """
        self.__file = open(path, file_mode)
        self._data_buffer = []
        if header is not None:
            self._data_buffer.append(header)
        self._separator = separator
        self._iteration = iteration

    def save(self):
        """Save the data on the file.

        Saves the data that the buffer has, in the file. Than
        the buffer is cleaned.
        """
        self.__file.write("".join(self._data_buffer))
        self._data_buffer = []

    def close_file(self):
        """Close the file.

        If there is data in the buffer, saves it and than closes the file.
        """
        if len(self._data_buffer) > 0:
            self.save()
        self.__file.close()