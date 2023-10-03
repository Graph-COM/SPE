from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import MessagePassing

from src_zinc.mlp import MLP


class GINE(nn.Module):
    layers: nn.ModuleList

    def __init__(
        self, n_layers: int, n_edge_types: int, in_dims: int, hidden_dims: int, out_dims: int,
        create_mlp: Callable[[int, int], MLP], bn: bool = False, residual: bool = False
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList()
        self.residual = residual
        self.bn = bn
        if bn:
            self.batch_norms = nn.ModuleList()
        for _ in range(n_layers - 1):
            layer = GINELayer(n_edge_types, in_dims, hidden_dims, create_mlp)
            self.layers.append(layer)
            in_dims = hidden_dims
            if bn:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dims))


        layer = GINELayer(n_edge_types, hidden_dims, out_dims, create_mlp)
        self.layers.append(layer)

    def forward(self, X_n: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor,
                PE: torch.Tensor) -> torch.Tensor:
        """
        :param X_n: Node feature matrix. [N_sum, D_in]
        :param edge_index: Graph connectivity in COO format. [2, E_sum]
        :param edge_attr: Edge type matrix. [E_sum]
        :return: Output node feature matrix. [N_sum, D_out]
        """
        for i, layer in enumerate(self.layers):
            X_0 = X_n
            X_n = layer(X_n, edge_index, edge_attr, PE)  # [N_sum, D_hid] or [N_sum, D_out]
            # batch normalization
            if self.bn and i < len(self.layers) - 1:
                if X_n.ndim == 3:
                    X_n = self.batch_norms[i](X_n.transpose(2, 1)).transpose(2, 1)
                else:
                    X_n = self.batch_norms[i](X_n)
            if self.residual:
                X_n = X_n + X_0
        return X_n


class GINELayer(MessagePassing):
    edge_features: nn.Embedding
    eps: nn.Parameter
    mlp: MLP

    def __init__(self, n_edge_types: int, in_dims: int, out_dims: int, create_mlp: Callable[[int, int], MLP]) -> None:
        # Use node_dim=0 because message() output has shape [E_sum, D_in] - https://stackoverflow.com/a/68931962
        super().__init__(aggr="add", flow="source_to_target", node_dim=0)
        # super(GINELayer, self).__init__(aggr='add')

        self.edge_features = nn.Embedding(n_edge_types, in_dims)
        #self.pe_embedding = nn.Linear(1, in_dims)
        self.pe_embedding = create_mlp(1, in_dims)
        self.eps = torch.nn.Parameter(data=torch.randn(1), requires_grad=True)
        self.mlp = create_mlp(in_dims, out_dims)

    def forward(self, X_n: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor,
                PE: torch.Tensor) -> torch.Tensor:
        """
        :param X_n: Node feature matrix. [N_sum, D_in]
        :param edge_index: Graph connectivity in COO format. [2, E_sum]
        :param edge_attr: Edge type matrix. [E_sum]
        :return: Output node feature matrix. [N_sum, D_out]
        """
        X_e = self.edge_features(edge_attr-1)   # [E_sum, D_in]
        if PE is not None:
            X_e *= self.pe_embedding(PE)

        # Contains sum(j in N(i)) {message(j -> i)} for each node i.
        S = self.propagate(edge_index, X=X_n, X_e=X_e)   # [N_sum, D_in]

        Z = (1 + self.eps) * X_n   # [N_sum, D_in]
        Z = Z + S                  # [N_sum, D_in]
        return self.mlp(Z)         # [N_sum, D_out]

    def message(self, X_j: torch.Tensor, X_e: torch.Tensor) -> torch.Tensor:
        """
        :param X_j: Features of the edge sources. [E_sum, D_in]
        :param X_e: Edge feature matrix. [E_sum, D_in]
        :return: The messages ReLU(X_j + E_ij) for each edge (j -> i). [E_sum, D_in]
        """
        return F.relu(X_j + X_e)   # [E_sum, D_in]

