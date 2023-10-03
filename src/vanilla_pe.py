from typing import List, Callable

import torch
from torch import nn
from torch_geometric.utils import unbatch


class NoPE(nn.Module):
    phi: nn.Module
    psi_list: nn.ModuleList

    def __init__(self, pe_dim: int) -> None:
        super().__init__()
        self.out_dim = pe_dim

    def forward(
            self, Lambda: torch.Tensor, V: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        """
        :param Lambda: Eigenvalue vectors. [B, D_pe]
        :param V: Concatenated eigenvector matrices. [N_sum, D_pe]
        :param edge_index: Graph connectivity in COO format. [2, E_sum]
        :param batch: Batch index vector. [N_sum]
        :return: Positional encoding matrix. [N_sum, D_pe]
        """
        # for sanity check
        return 0 * V   # [N_sum, D_pe]

    @property
    def out_dims(self) -> int:
        return self.out_dim

class IdPE(nn.Module):
    phi: nn.Module
    psi_list: nn.ModuleList

    def __init__(self, pe_dim: int) -> None:
        super().__init__()
        self.out_dim = pe_dim

    def forward(
        self, Lambda: torch.Tensor, V: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        """
        :param Lambda: Eigenvalue vectors. [B, D_pe]
        :param V: Concatenated eigenvector matrices. [N_sum, D_pe]
        :param edge_index: Graph connectivity in COO format. [2, E_sum]
        :param batch: Batch index vector. [N_sum]
        :return: Positional encoding matrix. [N_sum, D_pe]
        """
        return V   # [N_sum, D_pe]

    @property
    def out_dims(self) -> int:
        return self.out_dim