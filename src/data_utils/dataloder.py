import torch
import random
import torch.utils.data


from collections.abc import Mapping
from typing import List, Optional, Sequence, Union

from torch.utils.data.dataloader import default_collate

from torch_geometric.data import Dataset
from torch_geometric.data.data import BaseData
from torch_geometric.data.datapipes import DatasetAdapter
from src.data_utils.batch import Batch


def shuffle_group_graphs(graphs):
    for g in graphs:
        random.shuffle(g)
    random.shuffle(graphs)


class SameSizeDataLoader:
    # dataloader that loads graphs with the same size in a batch

    def __init__(self, dataset, batch_size, shuffle=False):
        self.dataset = dataset
        self.curr = -1
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.group_graphs = self.group_same_size(dataset)
        if shuffle:
            shuffle_group_graphs(self.group_graphs)
        self.iter = self.batch_same_size(self.group_graphs, batch_size)

    def group_same_size(self, dataset):
        graphs = []
        size2ind = {}
        ind = 0
        # group same size
        for data in dataset:
            if data.num_nodes in size2ind:
                graphs[size2ind[data.num_nodes]].append(data)
            else:
                size2ind[data.num_nodes] = ind
                ind += 1
                graphs.append([data])
        return graphs

    def batch_same_size(self, graphs, batch_size):
        # batch same size
        batched_graphs = []
        for g in graphs:
            for i in range(0, len(g), batch_size):
                # if i + batch_size <= len(g):
                batched_graphs.append(Batch.from_data_list(g[i : i+batch_size])) # batch graphs into a large graph

        return batched_graphs

    def _reset(self, shuffle):
        self.curr = -1
        if shuffle:
            shuffle_group_graphs(self.group_graphs)
            self.iter = self.batch_same_size(self.group_graphs, self.batch_size)

    def __len__(self):
        return len(self.iter)

    def __next__(self):
        self.curr += 1
        if self.curr == self.__len__(): # reach the end of dataloader
            self._reset(self.shuffle)
            raise StopIteration
        batch = self.iter[self.curr]
        return batch

    def __iter__(self):
        return self



class Collater:
    def __init__(self, follow_batch, exclude_keys):
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def __call__(self, batch):
        elem = batch[0]
        if isinstance(elem, BaseData):
            # return Batch.from_data_list(batch, self.follow_batch,
                                        # self.exclude_keys)
            return Batch.from_data_list(batch) # use customized one
        elif isinstance(elem, torch.Tensor):
            return default_collate(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, str):
            return batch
        elif isinstance(elem, Mapping):
            return {key: self([data[key] for data in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
            return type(elem)(*(self(s) for s in zip(*batch)))
        elif isinstance(elem, Sequence) and not isinstance(elem, str):
            return [self(s) for s in zip(*batch)]

        raise TypeError(f'DataLoader found invalid type: {type(elem)}')

    def collate(self, batch):  # pragma: no cover
        # TODO Deprecated, remove soon.
        return self(batch)


class DataLoader(torch.utils.data.DataLoader):
    r"""A data loader which merges data objects from a
    :class:`torch_geometric.data.Dataset` to a mini-batch.
    Data objects can be either of type :class:`~torch_geometric.data.Data` or
    :class:`~torch_geometric.data.HeteroData`.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        follow_batch (List[str], optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`None`)
        exclude_keys (List[str], optional): Will exclude each key in the
            list. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`.
    """
    def __init__(
        self,
        dataset: Union[Dataset, Sequence[BaseData], DatasetAdapter],
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        # Remove for PyTorch Lightning:
        kwargs.pop('collate_fn', None)

        # Save for PyTorch Lightning < 1.6:
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=Collater(follow_batch, exclude_keys),
            **kwargs,
        )