import numpy as np

class SegmentTree():
    def __init__(self, size):
        self.index = 0
        self.size = size
        self.full = False
        self.tree = np.zeros((2 * size - 1, ), dtype=np.float32)
        self.data = np.array([None] * size)
        self.max = 1

    # Propagates value up tree given a tree index
    def _propagate(self, index, value):
        parent = (index - 1) // 2
        left, right = 2 * parent + 1, 2 * parent + 2
        self.tree[parent] = self.tree[left] + self.tree[right]
        if parent != 0:
            self._propagate(parent, value)

    def update(self, index, value):
        self.tree[index] = value
        self._propagate(index, value)
        self.max = max(value, self.max)

    def append(self, data, value):
        self.data[self.index] = data
        self.update(self.index + self.size - 1, value)
        self.index = (self.index + 1) % self.size
        self.full = self.full or self.index == 0
        self.max = max(value, self.max)

    # Searches for the location of a value in sum tree
    def _retrieve(self, index, value):
        left, right = 2 * index + 1, 2 * index + 2
        if left >= len(self.tree):
            return index
        elif value <= self.tree[left]:
            return self._retrieve(left, value)
        else:
            return self._retrieve(right, value - self.tree[left])

    # Searches for a value in sum tree and returns value, data index and tree index
    def find(self, value):
        index = self._retrieve(0, value)
        data_index = index - self.size + 1
        return (self.tree[index], data_index, index)

    # Returns data given a data index
    def get(self, data_index):
        return self.data[data_index % self.size]

    def total(self):
        return self.tree[0]

    def __len__(self):
        return len(self.data)