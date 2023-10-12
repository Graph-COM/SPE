import torch
from torch import nn
from typing import List, Callable
from src.ign import IGN2to1

class SignInvPe(nn.Module):
    # pe = rho(phi(V)+phi(-V))
    def __init__(self, phi: nn.Module, rho: nn.Module) -> None:
        super(SignInvPe, self).__init__()
        self.phi = phi
        self.rho = rho

    def forward(
            self, Lambda: torch.Tensor, V: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        x = V.unsqueeze(-1) # TO DO: incorporate eigenvalues
        x = self.phi(x, edge_index) + self.phi(-x, edge_index) # [N, D_pe, hidden_dims]
        x = x.reshape([x.shape[0], -1]) # [N, D_pe * hidden_dims]
        x = self.rho(x) # [N, D_pe]

        return x

    @property
    def out_dims(self) -> int:
        return self.rho.out_dims


class MaskedSignInvPe(nn.Module):
    # pe = rho(mask-sum(phi(V)+phi(-V)))
    def __init__(self, phi: nn.Module, rho: nn.Module) -> None:
        super(MaskedSignInvPe, self).__init__()
        self.phi = phi
        self.rho = rho

    def forward(
            self, Lambda: torch.Tensor, V: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        x = V.unsqueeze(-1)  # TO DO: incorporate eigenvalues
        x = self.phi(x, edge_index) + self.phi(-x, edge_index)  # [N, D_pe, hidden_dims]
        pe_dim, N = x.size(1), x.size(0)
        num_nodes = [torch.sum(batch == i) for i in range(batch[-1]+1)]
        a = torch.arange(0, pe_dim).to(x.device)
        mask = torch.cat([(a < num).unsqueeze(0).repeat([num, 1]) for num in num_nodes], dim=0) # -1 since excluding zero eigenvalue
        x = (x*mask.unsqueeze(-1)).sum(dim=1) # [N, hidden_dims]
        x = self.rho(x)  # [N, D_pe]
        return x


    @property
    def out_dims(self) -> int:
        return self.rho.out_dims


class BasisInvPE(nn.Module):
    # pe = rho(phi(VV^{\top})), where phi is IGN
    def __init__(self, phi: nn.Module, rho: nn.Module) -> None:
        super(BasisInvPE, self).__init__()
        self.phi = phi
        self.rho = rho

    def forward(self, Lambda: torch.Tensor, P: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor):
        eig_feats_list = []
        pe_dim = Lambda.size(-1)
        for i, same_size_projs in enumerate(P):
            N = same_size_projs[list(same_size_projs.keys())[0]].size(-1)
            phi_outs = [self.phi(projs, mult) for mult, projs in same_size_projs.items()]
            eig_feats = torch.cat([phi_out.reshape(N, -1) for phi_out in phi_outs], dim=-1) # [N, min(N, pe_dim)]
            eig_feats = torch.cat([eig_feats, torch.zeros([N, pe_dim - torch.min(torch.tensor([N, pe_dim])).item()]).
                                  to(eig_feats.device)], dim=-1)  # [N, pe_dim]
            eig_feats = torch.cat((eig_feats, Lambda[i].unsqueeze(0).repeat(N, 1)), dim=-1)
            eig_feats_list.append(eig_feats)
        eig_feats = torch.cat(eig_feats_list, dim=0)
        return self.rho(eig_feats, edge_index)

    @property
    def out_dims(self) -> int:
        return self.rho.out_dims

class IGNBasisInv(nn.Module):
    """ IGN based basis invariant neural network
    """
    def __init__(self, mult_lst, in_channels, hidden_channels=16, num_layers=2, device='cuda', **kwargs):
        super(IGNBasisInv, self).__init__()
        self.encs = nn.ModuleList()
        self.mult_to_idx = {}
        curr_idx = 0
        for mult in mult_lst:
            # get a phi for each choice of multiplicity
            self.encs.append(IGN2to1(1, hidden_channels, mult, num_layers=num_layers, device=device))
            self.mult_to_idx[mult] = curr_idx
            curr_idx += 1

    def forward(self, proj, mult):
        enc_idx = self.mult_to_idx[mult]
        x = self.encs[enc_idx](proj)
        return x

