import torch
from torch_geometric.data import Data
from torch_geometric.utils import get_laplacian, to_dense_adj

class TargetNormalizer:
    def __init__(self, Y):
        self.mean = Y.mean(dim=0)
        self.std = Y.std(dim=0)

    def transform(self, dataset):
        # target normalization
        dataset.data.y = (dataset.data.y - self.mean) / self.std

