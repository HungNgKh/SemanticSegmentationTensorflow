from __future__ import division

import numpy as np
from abc import ABCMeta, abstractmethod


def data_augment(batch, methods):
    for method in methods:
        batch = method(batch)
    return batch


class Batch:


    def __init__(self, x , y):
        self.x = x
        self.y = y
        self.size = np.size(x, 0)



class DataSet:

    __metaclass__ = ABCMeta

    def __init__(self):
        self._current_index = 0
        self.size = 0

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def batch(self, batch_size):
        pass

    @abstractmethod
    def shuffle(self):
        pass

    def reset_index(self):
        self._current_index = 0


# Data set structure
class LoadTimeDataSet(DataSet):

    __metaclass__ = ABCMeta


    def __init__(self):
        super(LoadTimeDataSet, self).__init__()
        self.images = []
        self.labels = []


    # load data contain images and labels
    @abstractmethod
    def load(self):
        pass


    def shuffle(self):
        dataset = zip(self.images, self.labels)
        np.random.shuffle(dataset)
        self.images, self.labels = zip(*dataset)


    def batch(self, batch_size):

        index_end = self._current_index + batch_size
        if( index_end >= self.size):
            index_end = self.size
            batch = Batch(self.images[self._current_index : index_end], self.labels[self._current_index : index_end])
            self._current_index = 0
            return batch, True
        else:
            batch = Batch(self.images[self._current_index: index_end], self.labels[self._current_index : index_end])
            self._current_index = index_end
            return batch, False



class RunTimeDataSet(DataSet):

    __metaclass__ = ABCMeta

    def __init__(self):
        super(RunTimeDataSet, self).__init__()
        self._index_list = []

    @abstractmethod
    def load(self):
        pass

    def shuffle(self):
        np.random.shuffle(self._index_list)

    @abstractmethod
    def batch(self, batch_size):
        pass