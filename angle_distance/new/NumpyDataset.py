from Dataset import Dataset
import numpy as np

class NumpyDataset(Dataset):

    def __init__(self, dataset_path, batch_size, input_number, output_number, separator, lines_ignore):
        super(NumpyDataset, self).__init__(dataset_path, batch_size, input_number, output_number, separator, lines_ignore)
        self._load_dataset()


    def get_next(self):
        self._actual_index += 1
        data = self._dataset[self._actual_index-1]
        inputs = []
        outputs = []
        for (x, y) in data:
            inputs.append(x)
            outputs.append(y)
        return [inputs, outputs]
    
    def shuffle(self):
        np.random.shuffle(self._dataset)

    def _load_dataset(self):
        lines = np.genfromtxt(self._dataset_path, delimiter=self._separator)[self._lines_ignore:]
        self._size = len(lines)
        self._num_of_batches = self._size//self._batch_size
        if self._size%self._batch_size > 0:
            self._num_of_batches += 1
        self._dataset = []
        output_index = self._input_number+self._output_number
        for index in range(self._num_of_batches):
            data = lines[index*self._batch_size:(index+1)*self._batch_size]
            data = list(map(lambda line: ( line[:self._input_number], line[self._input_number:output_index] ), data))
            self._dataset.append(data)
        self._dataset = np.array(self._dataset)