"""LogString class. Logs variables into csv file."""
from Log import Log


class LogString(Log):
    """Logs variables into csv file."""

    def __init__(self, path, file_mode, header=None, separator=",", iteration=0 ):
        """Initialize variables.

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
        super(LogString, self).__init__(path, file_mode, header, separator, iteration)
    
    def log_string(self, variables):  # variables is a list
        """Add variables strings to the buffer.

        Recibes a list with variables, convert it to string and joins them
        with the separator attribute. Than stores the string in the buffer.

        Args:
            variables: a list of variables.
        """
        self._data_buffer.append(str(self._iteration)+self._separator +
                                 self._separator.join(map(str, variables))+
                                 "\n")
        self._iteration += 1
