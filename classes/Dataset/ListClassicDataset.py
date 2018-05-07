from ClassicDataset import ClassicDataset
import random

class ListClassicDataset(ClassicDataset):

    def __init__(self, dataset_path, batch_size, input_number, output_number, separator, lines_ignore, take=None):
        super(ListClassicDataset, self).__init__(dataset_path, batch_size, input_number, output_number, separator, lines_ignore, take)
        self._load_dataset()
    
    
    def get_next(self): 
        self._actual_index += 1
        data = self._dataset[self._actual_index-1][:]
        inputs = []
        outputs = []
        for (x, y) in data:
            inputs.append(x)
            outputs.append(y)
        return [inputs, outputs]

    
    def shuffle(self): 
        random.shuffle(self._dataset)


    def _load_dataset(self):
        lines = []
        with open(self._dataset_path, "r") as f:
            if self._take is not None:
                lines = [next(f) for line in range( (self._take+self._lines_ignore) )]
                lines = lines[self._lines_ignore:]
            else:
                lines = f.readlines()[self._lines_ignore:]
        self._dataset = []
        self._size = len(lines)
        self._total_batches = self._size//self._batch_size
        if self._size%self._batch_size > 0:
            self._total_batches += 1

        output_index = self._input_number+self._output_number
        for index in range(self._total_batches):
            data = lines[index*self._batch_size:(index+1)*self._batch_size]
            data = map(lambda line: [float(i) for i in line.split(self._separator) if i], data)
            data = list(map(lambda line: ( line[:self._input_number], line[self._input_number:output_index] ), data))
            self._dataset.append(data)