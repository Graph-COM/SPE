from typing import List, Callable

import torch
from torch import nn
from torch_geometric.utils import unbatch

from src_zinc.gin import GIN
from src_zinc.gine import GINE
from src_zinc.gin_deepsets import GINDeepsets
from src_zinc.ppgn import MaskedPPGN
from src_zinc.deepsets import DeepSets, MaskedDeepSets
from src_zinc.transformer import Transformer
from src_zinc.mlp import MLP
from src_zinc.utils import mask2d_sum_pooling, mask2d_diag_offdiag_meanpool

from src_zinc.schema import Schema

class StableExpressivePE(nn.Module):
    phi: nn.Module
    psi_list: nn.ModuleList

    def __init__(self, phi: nn.Module, psi_list: List[nn.Module]) -> None:
        super().__init__()
        self.phi = phi
        self.psi_list = nn.ModuleList(psi_list)

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
        Lambda = Lambda.unsqueeze(dim=2)   # [B, D_pe, 1]
        #Lambda = torch.cat([torch.cat([torch.cos(Lambda / 10000**(i/37)), torch.sin(Lambda / 10000**(i / 37))], dim=-1)
                            # for i in range(37)], dim=-1)
        Z = torch.stack([
            psi(Lambda).squeeze(dim=2)     # [B, D_pe]
            for psi in self.psi_list
        ], dim=2)                          # [B, D_pe, M]

        V_list = unbatch(V, batch, dim=0)   # [N_i, D_pe] * B
        Z_list = list(Z)                    # [D_pe, M] * B

        W_list = []                        # [N_i, N_i, M] * B
        for V, Z in zip(V_list, Z_list):   # [N_i, D_pe] and [D_pe, M]
            V = V.unsqueeze(dim=0)         # [1, N_i, D_pe]
            Z = Z.permute(1, 0)            # [M, D_pe]
            Z = Z.diag_embed()             # [M, D_pe, D_pe]
            V_T = V.mT                     # [1, D_pe, N_i]
            W = V.matmul(Z).matmul(V_T)    # [M, N_i, N_i]
            # W = V.matmul(V_T).repeat([Z.size(0), 1, 1])
            W = W.permute(1, 2, 0)         # [N_i, N_i, M]
            W_list.append(W)

        return self.phi(W_list, edge_index)   # [N_sum, D_pe]

    @property
    def out_dims(self) -> int:
        return self.phi.out_dims


class MaskedStableExpressivePE(nn.Module):
    phi: nn.Module
    psi_list: nn.ModuleList

    def __init__(self, phi: nn.Module, psi_list: List[nn.Module]) -> None:
        super().__init__()
        self.phi = phi
        self.psi_list = nn.ModuleList(psi_list)

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
        Lambda = Lambda.unsqueeze(dim=2)   # [B, D_pe, 1]
        # Lambda = torch.cat([torch.cat([torch.cos(Lambda / 10000**(i/8)), torch.sin(Lambda / 10000**(i / 8))], dim=-1)
                            # for i in range(16)], dim=-1)
        a = torch.arange(0, Lambda.size(1)).unsqueeze(0).to(Lambda.device)
        mask = torch.cat([a < torch.sum(batch == i) for i in range(batch[-1]+1)], dim=0) # [B, D_pe, 1]
        Z = torch.stack([
            psi(Lambda, mask.unsqueeze(-1)).squeeze(dim=2)     # [B, D_pe]
            for psi in self.psi_list
        ], dim=2)                          # [B, D_pe, M]

        V_list = unbatch(V, batch, dim=0)   # [N_i, D_pe] * B
        Z_list = list(Z)                    # [D_pe, M] * B

        W_list = []                        # [N_i, N_i, M] * B
        for V, Z in zip(V_list, Z_list):   # [N_i, D_pe] and [D_pe, M]
            V = V.unsqueeze(dim=0)         # [1, N_i, D_pe]
            Z = Z.permute(1, 0)            # [M, D_pe]
            Z = Z.diag_embed()             # [M, D_pe, D_pe]
            V_T = V.mT                     # [1, D_pe, N_i]
            W = V.matmul(Z).matmul(V_T)    # [M, N_i, N_i]
            W = W.permute(1, 2, 0)         # [N_i, N_i, M]
            W_list.append(W)

        return self.phi(W_list, edge_index)   # [N_sum, D_pe]

    @property
    def out_dims(self) -> int:
        return self.phi.out_dims


class MLPPhi(nn.Module):
    gin: GIN

    def __init__(
            self, n_layers: int, in_dims: int, hidden_dims: int, out_dims: int, create_mlp: Callable[[int, int], MLP]
    ) -> None:
        super().__init__()
        self.mlp = MLP(n_layers, in_dims, hidden_dims, out_dims, use_bn=False, activation='relu', dropout_prob=0.0)

    def forward(self, W_list: List[torch.Tensor], edge_index: torch.Tensor) -> torch.Tensor:
        """
        :param W_list: The {V * psi_l(Lambda) * V^T: l in [m]} tensors. [N_i, N_i, M] * B
        :param edge_index: Graph connectivity in COO format. [2, E_sum]
        :return: Positional encoding matrix. [N_sum, D_pe]
        """
        n_max = max(W.size(0) for W in W_list)
        W_pad_list = []     # [N_i, N_max, M] * B
        mask = [] # node masking, [N_i, N_max] * B
        for W in W_list:
            zeros = torch.zeros(W.size(0), n_max - W.size(1), W.size(2), device=W.device)
            W_pad = torch.cat([W, zeros], dim=1)   # [N_i, N_max, M]
            W_pad_list.append(W_pad)
            mask.append((torch.arange(n_max, device=W.device) < W.size(0)).tile((W.size(0), 1))) # [N_i, N_max]

        W = torch.cat(W_pad_list, dim=0)   # [N_sum, N_max, M]
        mask = torch.cat(mask, dim=0)   # [N_sum, N_max]
        PE = self.mlp(W)       # [N_sum, N_max, D_pe]
        return (PE * mask.unsqueeze(-1)).sum(dim=1)               # [N_sum, D_pe]
        # return PE.sum(dim=1)

    @property
    def out_dims(self) -> int:
        return self.mlp.out_dims



class GINPhi(nn.Module):
    gin: GIN

    def __init__(
        self, n_layers: int, in_dims: int, hidden_dims: int, out_dims: int, create_mlp: Callable[[int, int], MLP], bn: bool
    ) -> None:
        super().__init__()
        self.gin = GIN(n_layers, in_dims, hidden_dims, out_dims, create_mlp, bn)
        self.mlp = create_mlp(out_dims, out_dims)

    def forward(self, W_list: List[torch.Tensor], edge_index: torch.Tensor) -> torch.Tensor:
        """
        :param W_list: The {V * psi_l(Lambda) * V^T: l in [m]} tensors. [N_i, N_i, M] * B
        :param edge_index: Graph connectivity in COO format. [2, E_sum]
        :return: Positional encoding matrix. [N_sum, D_pe]
        """
        n_max = max(W.size(0) for W in W_list)
        W_pad_list = []     # [N_i, N_max, M] * B
        mask = [] # node masking, [N_i, N_max] * B
        for W in W_list:
            zeros = torch.zeros(W.size(0), n_max - W.size(1), W.size(2), device=W.device)
            W_pad = torch.cat([W, zeros], dim=1)   # [N_i, N_max, M]
            W_pad_list.append(W_pad)
            mask.append((torch.arange(n_max, device=W.device) < W.size(0)).tile((W.size(0), 1))) # [N_i, N_max]

        W = torch.cat(W_pad_list, dim=0)   # [N_sum, N_max, M]
        mask = torch.cat(mask, dim=0)   # [N_sum, N_max]
        PE = self.gin(W, edge_index)       # [N_sum, N_max, D_pe]
        PE = (PE * mask.unsqueeze(-1)).sum(dim=1)
        return PE               # [N_sum, D_pe]
        # return PE.sum(dim=1)

    @property
    def out_dims(self) -> int:
        return self.gin.out_dims


class GINEPhi(nn.Module):
    gine: GINE

    def __init__(
            self, n_layers: int, in_dims: int, hidden_dims: int, out_dims: int, create_mlp: Callable[[int, int], MLP]
    ) -> None:
        super().__init__()
        self.gine = GINE(n_layers, in_dims, hidden_dims, out_dims, create_mlp)

    def forward(self, W_list: List[torch.Tensor], edge_index: torch.Tensor) -> torch.Tensor:
        """
        :param W_list: The {V * psi_l(Lambda) * V^T: l in [m]} tensors. [N_i, N_i, M] * B
        :param edge_index: Graph connectivity in COO format. [2, E_sum]
        :return: Positional encoding matrix. [N_sum, D_pe]
        """
        n_max = max(W.size(0) for W in W_list)
        W_pad_list = []                            # [N_i, N_max, M] * B
        for W in W_list:
            zeros = torch.zeros(W.size(0), n_max - W.size(1), W.size(2), device=W.device)
            W_pad = torch.cat([W, zeros], dim=1)   # [N_i, N_max, M]
            W_pad_list.append(W_pad)

        W = torch.cat(W_pad_list, dim=0)   # [N_sum, N_max, M]
        PE = self.gin(W, edge_index)       # [N_sum, N_max, D_pe]
        return PE.sum(dim=1)               # [N_sum, D_pe]

    @property
    def out_dims(self) -> int:
        return self.gin.out_dims


class PPGNPhi(nn.Module):
    ppgn: MaskedPPGN

    def __init__(self, n_layers: int, in_dims: int, hidden_dims: int, out_dims: int,
                 create_mlp: Callable[[int, int], MLP]) -> None:
        super(PPGNPhi, self).__init__()
        self.ppgn = MaskedPPGN(in_dims, hidden_dims, out_dims, create_mlp, num_rb_layer=n_layers)
        self.pe_project = nn.Linear(2*out_dims, out_dims)

    def forward(self, W_list: List[torch.Tensor], edge_index: torch.Tensor) -> torch.Tensor:
        """
        :param W_list: The {V * psi_l(Lambda) * V^T: l in [m]} tensors. [N_i, N_i, M] * B
        :param edge_index: Graph connectivity in COO format. [2, E_sum]
        :return: Positional encoding matrix. [N_sum, D_pe]
        """
        # No edge info incorporated currently, TO DO: incorporate edge info into W
        n_max = max(W.size(0) for W in W_list)
        W_pad_list = []  # [N_max, N_max, M] * B
        mask = []
        for W in W_list:
            zeros = torch.zeros(W.size(0), n_max - W.size(1), W.size(2), device=W.device)
            W_pad = torch.cat([W, zeros], dim=1)  # [N_i, N_max, M]
            zeros = torch.zeros(n_max - W_pad.size(0), W_pad.size(1), W_pad.size(2), device=W_pad.device)
            W_pad = torch.cat([W_pad, zeros], dim=0)  # [N_max, N_max, M]
            W_pad = torch.unsqueeze(W_pad, dim=0) # [1, N_max, N_max, M]
            W_pad_list.append(W_pad)
            mask.append((torch.arange(n_max, device=W.device) < W.size(0)).unsqueeze(0)) # [1, N_max]

        W = torch.cat(W_pad_list, dim=0)  # [B, N_max, N_max, M]
        mask = torch.cat(mask, dim=0) # [B, N_max]
        mask_2d = mask.float().unsqueeze(-1) # [B, N_max, 1]
        mask_2d = torch.matmul(mask_2d, mask_2d.transpose(1, 2)).unsqueeze(-1) # [B, N_max, N_max, 1]
        PE = self.ppgn(W, mask_2d)   # [B, N_max, N_max, D_pe]
        # PE = mask2d_sum_pooling(PE, mask_2d) # TO DO: more variants of pooling functions, e.g. diag/off-diag pooling
        PE = mask2d_diag_offdiag_meanpool(PE, mask_2d)
        PE = self.pe_project(PE)
        # PE = PE.sum(dim=1)
        PE = PE.view(-1, PE.size(-1))[mask.view(-1)] # [N_sum, D_pe]
        return PE

    @property
    def out_dims(self) -> int:
        return self.ppgn.out_dims


class GINDeepSetsPhi(nn.Module):
    """
    inspired by Vignac, Clement, Andreas Loukas, and Pascal Frossard.
    "Building powerful and equivariant graph neural networks with structural message-passing."
    Advances in neural information processing systems 33 (2020): 14143-14155.
    """
    def __init__(self, n_layers: int, in_dims: int, hidden_dims: int, out_dims: int,
                 create_mlp: Callable[[int, int], MLP]):
        super(GINDeepSetsPhi, self).__init__()
        self.gin_deepsets = GINDeepsets(n_layers, in_dims, hidden_dims, out_dims, create_mlp)

    def forward(self, W_list: List[torch.Tensor], edge_index: torch.Tensor) -> torch.Tensor:
        """
        :param W_list: The {V * psi_l(Lambda) * V^T: l in [m]} tensors. [N_i, N_i, M] * B
        :param edge_index: Graph connectivity in COO format. [2, E_sum]
        :return: Positional encoding matrix. [N_sum, D_pe]
        """
        n_max = max(W.size(0) for W in W_list)
        W_pad_list = []     # [N_i, N_max, M] * B
        mask = [] # node masking, [N_i, N_max] * B
        for W in W_list:
            zeros = torch.zeros(W.size(0), n_max - W.size(1), W.size(2), device=W.device)
            W_pad = torch.cat([W, zeros], dim=1)   # [N_i, N_max, M]
            W_pad_list.append(W_pad)
            mask.append((torch.arange(n_max, device=W.device) < W.size(0)).tile((W.size(0), 1))) # [N_i, N_max]

        W = torch.cat(W_pad_list, dim=0)   # [N_sum, N_max, M]
        mask = torch.cat(mask, dim=0).unsqueeze(-1)   # [N_sum, N_max]
        PE = self.gin_deepsets(W, edge_index, mask)       # [N_sum, N_max, D_pe]
        return (PE * mask).sum(dim=1)               # [N_sum, D_pe]

    @property
    def out_dims(self) -> int:
        return self.gin_deepsets.out_dims


def GetPhi(cfg: Schema, create_mlp: Callable[[int, int], MLP]):
    if cfg.phi_model_name == 'gin':
        return GINPhi(cfg.n_phi_layers, cfg.n_psis, cfg.phi_hidden_dims, cfg.pe_dims,
                                         create_mlp, cfg.batch_norm)
        #return GINPhi(cfg.n_phi_layers, cfg.n_psis, cfg.phi_hidden_dims, cfg.pe_dims,
                      #create_mlp) # no bn for padding input
    elif cfg.phi_model_name == 'gin_deepsets':
        return GINDeepSetsPhi(cfg.n_phi_layers, cfg.n_psis, cfg.phi_hidden_dims, cfg.pe_dims,
                                         create_mlp)
    elif cfg.phi_model_name == 'ppgn': # unstable now
        return PPGNPhi(cfg.n_phi_layers, cfg.n_psis, cfg.phi_hidden_dims, cfg.pe_dims,
                                         create_mlp)
    elif cfg.phi_model_name == 'mlp':
        return MLPPhi(cfg.n_phi_layers, cfg.n_psis, cfg.phi_hidden_dims, cfg.pe_dims, create_mlp)
    else:
        raise Exception ("Phi function not implemented!")


def GetPsi(cfg: Schema):
    if cfg.psi_model_name == 'deepsets':
        if cfg.pe_method.startswith("masked"):
            return MaskedDeepSets(cfg.n_psi_layers, 1, cfg.psi_hidden_dims, 1, cfg.psi_activation)
        else:
            return DeepSets(cfg.n_psi_layers, 1, cfg.psi_hidden_dims, 1, cfg.psi_activation)
    elif cfg.psi_model_name == 'transformer':
        return Transformer(cfg.n_psi_layers, 1, cfg.psi_hidden_dims, 1, cfg.num_heads)
    elif cfg.psi_model_name == 'mlp':
        return MLP(cfg.n_psi_layers, 1, cfg.psi_hidden_dims, 1, use_bn=cfg.mlp_use_bn, activation=cfg.psi_activation,
                   dropout_prob=0.0)
    else:
        raise Exception ('Psi function not implemented!')
