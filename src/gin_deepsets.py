from typing import Callable

import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from src.deepsets import MaskedDeepSetsLayer
from src.mlp import MLP

class GINDeepsets(nn.Module):
    layers: nn.ModuleList

    def __init__(
            self, n_layers: int, in_dims: int, hidden_dims: int, out_dims: int, create_mlp: Callable[[int, int], MLP]
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList()
        for _ in range(n_layers - 1):
            layer = GINDeepsetsLayer(create_mlp(in_dims, hidden_dims), activation='relu')
            self.layers.append(layer)
            in_dims = hidden_dims

        layer = GINDeepsetsLayer(create_mlp(hidden_dims, out_dims), activation='id') # drop last activation
        self.layers.append(layer)

    def forward(self, X: torch.Tensor, edge_index: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        :param X: Node feature matrix. [N_sum, N_max, D_in]
        :param edge_index: Graph connectivity in COO format. [2, E_sum]
        :param mask: fake node masking. [N_sum, N_max, 1]
        :return: Output node feature matrix. [N_sum, N_max, D_out]
        """
        for layer in self.layers:
            X = layer(X, edge_index, mask)   # [N_sum, ***, D_hid] or [N_sum, ***, D_out]
        return X                       # [N_sum, ***, D_out]

    @property
    def out_dims(self) -> int:
        return self.layers[-1].out_dims


class GINDeepsetsLayer(MessagePassing):
    eps: nn.Parameter
    mlp: MLP

    def __init__(self, mlp: MLP, activation: str) -> None:
        # Use node_dim=0 because message() output has shape [E_sum, ***, D_in] - https://stackoverflow.com/a/68931962
        super().__init__(aggr="add", flow="source_to_target", node_dim=0)

        self.eps = torch.nn.Parameter(data=torch.randn(1), requires_grad=True)
        self.mlp = mlp
        self.maksed_deepset_layer = MaskedDeepSetsLayer(mlp.out_dims, mlp.out_dims, activation)

    def forward(self, X: torch.Tensor, edge_index: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        :param X: Node feature matrix. [N_sum, N_max, D_in]
        :param edge_index: Graph connectivity in COO format. [2, E_sum]
        :param mask: fake node masking. [N_sum, N_max, 1]
        :return: Output node feature matrix. [N_sum, N_max, D_out]
        """
        # Contains sum(j in N(i)) {message(j -> i)} for each node i.


        S = self.propagate(edge_index, X=X)   # [N_sum, N_max, D_in]

        Z = (1 + self.eps) * X   # [N_sum, N_max, D_in]
        Z = Z + S                # [N_sum, N_max, D_in]
        Z = self.mlp(Z) # [N_sum, N_max, D_out]

        # masked deepset layer
        return self.maksed_deepset_layer(Z, mask)       # [N_sum, N_max, D_out]

    def message(self, X_j: torch.Tensor) -> torch.Tensor:
        """
        :param X_j: Features of the edge sources. [E_sum, ***, D_in]
        :return: The messages X_j for each edge (j -> i). [E_sum, ***, D_in]
        """
        return X_j   # [E_sum, ***, D_in]

    @property
    def out_dims(self) -> int:
        return self.mlp.out_dims
