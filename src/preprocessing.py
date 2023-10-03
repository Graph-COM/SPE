import torch
from torch_geometric.data import Data
from torch_geometric.utils import get_laplacian, to_dense_adj

class TargetNormalizer:
    def __init__(self, Y):
        self.mean = Y.mean()
        self.std = Y.std()

    def transform(self, dataset):
        # target normalization
        dataset._data.y = (dataset.y - self.mean) / self.std

