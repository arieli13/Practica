from IncrementalDataset import IncrementalDataset
import random

class ListIncrementalDataset(IncrementalDataset):
    
    def __init__(self, dataset_path, increment_size, input_number, output_number, separator, lines_ignore, buffer_size, take=None, repeat=1):
        super(ListIncrementalDataset, self).__init__(dataset_path, increment_size, input_number, output_number, separator, lines_ignore, buffer_size, take)
        self._load_dataset()
        self._repeat = repeat
        self.__repeat()
        self.increment_dataset()
    
    def increment_dataset(self):
        for index in range(self._increment_size):
            self._dataset.append(self._full_dataset.pop())
        self._size_full_dataset -= self._increment_size
        self._size += self._increment_size
        if self._size > self._buffer_size:
            self._dataset = self._dataset[self._size-self._buffer_size:]
            self._size = self._buffer_size
        self._increments_done += 1
    
    def get_next(self): 
        self._actual_index += 1
        data = self._dataset[self._actual_index-1][:]
        inputs = []
        outputs = []
        for (x, y) in data:
            inputs.append(x)
            outputs.append(y)
        return [inputs, outputs]
    
    def _load_dataset(self):
        lines = []
        with open(self._dataset_path, "r") as f:
            if self._take is not None:
                lines = [next(f) for line in range( (self._take+self._lines_ignore) )]
                lines = lines[self._lines_ignore:]
            else:
                lines = f.readlines()[self._lines_ignore:]
        self._dataset = []
        self._full_dataset = []

        self._size_full_dataset = len(lines)

        output_index = self._input_number+self._output_number
        for index in range(self._size_full_dataset):
            data = lines[index:(index+1)]
            data = map(lambda line: [float(i) for i in line.split(self._separator) if i], data)
            data = list(map(lambda line: ( line[:self._input_number], line[self._input_number:output_index] ), data))
            self._full_dataset.append(data)
        self._num_of_increments = self._size_full_dataset//self._increment_size
    
    def shuffle(self):
        random.shuffle(self._dataset)
    
    def __repeat(self):
        self._full_dataset = self._full_dataset*self._repeat
        self._size_full_dataset *= self._repeat
        self._num_of_increments *= self._repeat
