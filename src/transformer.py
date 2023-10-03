import torch
from torch import nn
from torch.nn import MultiheadAttention

class Transformer(nn.Module):
    def __init__(self, n_layers: int, in_dims: int, hidden_dims: int, out_dims: int, num_heads: int) -> None:
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList()
        self.linear_in = nn.Linear(in_dims, hidden_dims)
        self.linear_out = nn.Linear(hidden_dims, out_dims)
        for _ in range(n_layers):
            layer = MultiheadAttention(hidden_dims, num_heads, batch_first=True)
            self.layers.append(layer)


    def forward(self,X: torch.Tensor) -> torch.Tensor:
        X = self.linear_in(X)
        for layer in self.layers:
            X, _ = layer(X, X, X) # self-attention
        return self.linear_out(X)
