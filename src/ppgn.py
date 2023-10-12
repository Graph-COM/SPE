import torch
from torch import nn
from src.mlp import MLP
from typing import Callable

class MaskedPPGN(nn.Module):
    # Implementation of PPGN, as described in Section 7 in
    # Maron, Haggai, et al. "Provably powerful graph networks." Advances in neural information processing systems 32 (2019).
    # Instead of grouping the equal-size data in the same batch, we use a masking method

    def __init__(self, in_dims: int, hidden_dims: int, out_dims: int, create_mlp: Callable[[int, int], MLP],
                 num_rb_layer: int = 4):
        super(MaskedPPGN, self).__init__()
        # ppgn regular blocks
        self.ppgn_rb = torch.nn.ModuleList()
        for i in range(num_rb_layer - 1):
            self.ppgn_rb.append(RegularBlock(in_dims, hidden_dims, create_mlp))
            in_dims = hidden_dims
        self.ppgn_rb.append(RegularBlock(hidden_dims, out_dims, create_mlp))

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        # x: [B, N, N, in_dims], mask: [B, N, N, 1]
        for rb in self.ppgn_rb:
            x = rb(x, mask)
        return x

    @property
    def out_dims(self) -> int:
        return self.ppgn_rb[-1].out_dims


class RegularBlock(nn.Module):
    # input X: [B, N, N, in_dims], output = SkipConnection(X, m1(X) @ m2(X)): [B, N, N, out_dims]

    def __init__(self, in_dims: int, out_dims: int, create_mlp: Callable[[int, int], MLP]):
        super(RegularBlock, self).__init__()
        self.mlp1 = create_mlp(in_dims, out_dims)
        self.mlp2 = create_mlp(in_dims, out_dims)
        self.skip = SkipConnectionBlock(in_dims+out_dims, out_dims, create_mlp)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        # x: [B, N, N, in_dims], mask: [B, N, N, 1]
        mlp1 = mask * self.mlp1(x) # [B, N, N, out_dims], mask fake nodes
        mlp2 = mask * self.mlp2(x)

        mult = torch.matmul(mlp1.transpose(1, -1), mlp2.transpose(1, -1)) # [B, out_dims, N, N]
        mult = mult.transpose(1, -1) # [B, N, N, out_dims]

        out = self.skip(x, mult)
        return out

    @property
    def out_dims(self) -> int:
        return self.mlp1.out_dims


class SkipConnectionBlock(nn.Module):
    def __init__(self, in_dims: int, out_dims: int, create_mlp: Callable[[int, int], MLP]):
        super(SkipConnectionBlock, self).__init__()
        self.mlp = create_mlp(in_dims, out_dims)

    def forward(self, x0: torch.Tensor, x: torch.Tensor):
        # X0: [B, N, N, in_dims]
        # X:  [B, N, N, out_dims]
        out = torch.cat((x0, x), dim=-1)
        out = self.mlp(out)
        return out

