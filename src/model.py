from typing import Callable

import torch
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool, global_add_pool

from src.gine import GINE
from src.deepsets import DeepSets
from src.mlp import MLP
from src.stable_expressive_pe import StableExpressivePE, MaskedStableExpressivePE, GetPhi, GetPsi
from src.sign_inv_pe import SignInvPe, BasisInvPE, IGNBasisInv, MaskedSignInvPe
from src.vanilla_pe import IdPE
from src.gin import GIN
from src.pna import PNA
from src.schema import Schema


# def construct_model(cfg: Schema, create_mlp: Callable[[int, int], MLP], **kwargs):
def construct_model(cfg: Schema, list_create_mlp, **kwargs): # a list of mlp creators
    # final model = pe_method + base_model
    # base_model
    # two mlps using batch norm and layer norm
    create_mlp, create_mlp_ln = list_create_mlp if isinstance(list_create_mlp, tuple) else (list_create_mlp, list_create_mlp)
    target_dim = 1 if kwargs.get('target_dim') is None else kwargs.get('target_dim')
    if cfg.base_model == 'gine':
        base_model = GINEBaseModel(
            cfg.n_base_layers, cfg.n_edge_types, cfg.node_emb_dims, cfg.base_hidden_dims, create_mlp,
            residual=kwargs.get("residual"), feature_type=kwargs.get('feature_type'), pooling=cfg.pooling,
            target_dim=target_dim
        )
    elif cfg.base_model == 'pna':
        # cannot use now
        assert kwargs.get("deg") is not None
        base_model = PNABaseModel(cfg.n_base_layers, cfg.n_edge_types, cfg.node_emb_dims, cfg.base_hidden_dims,
                                  create_mlp, deg=kwargs.get("deg"), residual=kwargs.get("residual"),
                                  bn=kwargs.get("bn"), sn=kwargs.get("sn"))
    else:
        raise Exception("Base model not implemented!")
    # pe_method
    if cfg.pe_method == "sign_inv":
        # gin = GIN(cfg.n_phi_layers, 1, cfg.phi_hidden_dims, cfg.phi_hidden_dims, create_mlp, bn=cfg.batch_norm)  # 1=eigenvec
        gin = GIN(cfg.n_phi_layers, 1, cfg.phi_hidden_dims, 4, create_mlp, bn=cfg.batch_norm)  # 1=eigenvec
        # rho = create_mlp(cfg.pe_dims * cfg.phi_hidden_dims, cfg.pe_dims)
        rho = MLP(cfg.n_psi_layers, cfg.pe_dims * 4, cfg.phi_hidden_dims,
                  cfg.pe_dims, use_bn=cfg.mlp_use_bn, activation='relu', dropout_prob=0.0)
        return Model(
            cfg.n_node_types, cfg.node_emb_dims,
            positional_encoding=SignInvPe(phi=gin, rho=rho),
            base_model=base_model,
            pe_aggregate=cfg.pe_aggregate,
            feature_type=kwargs.get("feature_type")
        )
    elif cfg.pe_method == 'masked_sign_inv':
        gin = GIN(cfg.n_phi_layers, 1, cfg.phi_hidden_dims, cfg.phi_hidden_dims, create_mlp, bn=cfg.batch_norm)  # 1=eigenvec
        # rho = create_mlp(cfg.phi_hidden_dims, cfg.pe_dims)
        rho = MLP(cfg.n_psi_layers, cfg.phi_hidden_dims, cfg.phi_hidden_dims, cfg.pe_dims, use_bn=cfg.mlp_use_bn,
                  activation='relu', dropout_prob=0.0)
        return Model(
            cfg.n_node_types, cfg.node_emb_dims,
            positional_encoding=MaskedSignInvPe(phi=gin, rho=rho),
            base_model=base_model,
            pe_aggregate=cfg.pe_aggregate,
            feature_type=kwargs.get("feature_type")
        )
    elif cfg.pe_method == 'basis_inv':
        assert kwargs.get("uniq_mults") is not None
        uniq_mults = kwargs.get("uniq_mults")
        Phi = IGNBasisInv(uniq_mults, 1, hidden_channels=cfg.phi_hidden_dims, **kwargs)
        # rho = create_mlp(2 * cfg.pe_dims, cfg.pe_dims) # 2 * pe_dim = eigenvalues + eigenvectors
        rho = GIN(cfg.n_psi_layers, 2 * cfg.pe_dims, 2 * cfg.pe_dims, cfg.pe_dims, create_mlp)
        return Model(
            cfg.n_node_types, cfg.node_emb_dims,
            positional_encoding=BasisInvPE(phi=Phi, rho=rho),
            base_model=base_model,
            pe_aggregate = cfg.pe_aggregate,
            feature_type = kwargs.get("feature_type")
        )
    elif cfg.pe_method.endswith('spe'):
        Phi = GetPhi(cfg, create_mlp_ln, kwargs['device']) # for phi function, use layer norm
        Psi_list = [
            GetPsi(cfg)
            for _ in range(cfg.n_psis)
        ]
        # Stable PE
        if cfg.pe_method == 'spe':
            pe_model = StableExpressivePE(Phi, Psi_list)
        elif cfg.pe_method == 'masked_spe':
            pe_model = MaskedStableExpressivePE(Phi, Psi_list)
        return Model(
            cfg.n_node_types, cfg.node_emb_dims,
            positional_encoding=pe_model,
            base_model=base_model,
            pe_aggregate = cfg.pe_aggregate,
            feature_type = kwargs.get("feature_type")
        )
    elif cfg.pe_method == 'none':
        return Model(cfg.n_node_types, cfg.node_emb_dims, positional_encoding=None, base_model=base_model,
                     pe_aggregate=None, feature_type = kwargs.get("feature_type"))
    elif cfg.pe_method == 'id':
        return Model(cfg.n_node_types, cfg.node_emb_dims, positional_encoding=IdPE(cfg.pe_dims), base_model=base_model,
                     pe_aggregate=cfg.pe_aggregate, feature_type = kwargs.get("feature_type"))
    else:
        raise Exception("PE method not implemented!")


class Model(nn.Module):
    node_features: nn.Embedding
    positional_encoding: nn.Module
    fc: nn.Linear
    base_model: nn.Module

    def __init__(
        self, n_node_types: int, node_emb_dims: int, positional_encoding: nn.Module, base_model: nn.Module,
            pe_aggregate: str, feature_type: str = "discrete"
    ) -> None:
        super().__init__()

        self.node_features = nn.Embedding(n_node_types, node_emb_dims) if feature_type == "discrete" else \
                             nn.Linear(n_node_types, node_emb_dims)
        self.base_model = base_model
        self.positional_encoding = positional_encoding
        if positional_encoding is not None:
            self.pe_embedding = nn.Linear(self.positional_encoding.out_dims, node_emb_dims)
            self.pe_aggregate = pe_aggregate # "add" or "concat"
            assert pe_aggregate == "add" or pe_aggregate == "concat" or pe_aggregate == "peg"
            if pe_aggregate == "concat":
                self.fc = nn.Linear(2 * node_emb_dims, node_emb_dims, bias=True)
        # self.fc = nn.Linear(self.positional_encoding.out_dims, node_emb_dims, bias=True)

    def forward(self, batch: Batch) -> torch.Tensor:
        X_n = self.node_features(batch.x.squeeze(dim=1))    # [N_sum, D]
        PE = None
        if self.positional_encoding is not None:
            eig_mats = batch.P if "P" in batch else batch.V # pass projection matrices if using BasisNet
            PE = self.positional_encoding(batch.Lambda, eig_mats, batch.edge_index, batch.batch)   # [N_sum, D_pe]
            if self.pe_aggregate == "add":
                # PE = self.pe_embedding(PE)
                X_n = X_n + self.pe_embedding(PE)
                # PE = None
            elif self.pe_aggregate == "concat":
                # PE = self.pe_embedding(PE)
                X_n = torch.cat([X_n, self.pe_embedding(PE)], dim=-1)
                X_n = self.fc(X_n)                                                                    # [N_sum, D]
                # PE = None
            elif self.pe_aggregate == "peg":
                PE = torch.linalg.norm(PE[batch.edge_index[0]] - PE[batch.edge_index[1]], dim=-1)
                PE = PE.view([-1, 1])
        return self.base_model(X_n, batch.edge_index, batch.edge_attr, PE, batch.snorm if "snorm" in batch else None
                               , batch.batch)           # [B]



class GINEBaseModel(nn.Module):
    gine: GINE

    def __init__(
        self, n_layers: int, n_edge_types: int, in_dims: int, hidden_dims: int, create_mlp: Callable[[int, int], MLP],
            residual: bool = False, bn: bool = False, feature_type: str = "discrete", pooling: str = "mean",
            target_dim: int = 1) -> None:
        super().__init__()
        self.gine = GINE(n_layers, n_edge_types, in_dims, hidden_dims, hidden_dims, create_mlp, residual=residual,
                         bn=bn, feature_type=feature_type)
        self.mlp = create_mlp(hidden_dims, target_dim)
        self.pooling = global_mean_pool if pooling == 'mean' else global_add_pool

    def forward(
        self, X_n: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, PE: torch.Tensor, snorm: torch.Tensor,
            batch: torch.Tensor
    ) -> torch.Tensor:
        """
        :param X_n: Node feature matrix. [N_sum, D_in]
        :param edge_index: Graph connectivity in COO format. [2, E_sum]
        :param edge_attr: Edge type matrix. [E_sum]
        :param batch: Batch index vector. [N_sum]
        :return: Predicted regression values. [B]
        """
        X_n = self.gine(X_n, edge_index, edge_attr, PE)   # [N_sum, hidden_dim]
        # X_n = global_mean_pool(X_n, batch) # [B, hidden_dim]
        X_n = self.pooling(X_n, batch) # [B, hidden_dim]
        Y_pred = self.mlp(X_n)         # [B, 1]
        return Y_pred.squeeze(dim=1)                  # [B]


class PNABaseModel(nn.Module):
    def __init__(self, n_layers: int, n_edge_types: int, in_dims: int, hidden_dims: int,
                 create_mlp: Callable[[int, int], MLP], deg: torch.Tensor, residual: bool = False, bn: bool = False,
                 sn: bool = False
                 ) -> None:
        super(PNABaseModel, self).__init__()
        self.pna = PNA(n_layers, n_edge_types, in_dims, hidden_dims, create_mlp, deg=deg, residual=residual,
                       bn=bn, sn=sn)
        self.mlp = create_mlp(hidden_dims, 1)

    def forward(
            self, X_n: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, snorm: torch.Tensor,
            batch: torch.Tensor
    ) -> torch.Tensor:
        """
        :param X_n: Node feature matrix. [N_sum, D_in]
        :param edge_index: Graph connectivity in COO format. [2, E_sum]
        :param edge_attr: Edge type matrix. [E_sum]
        :param batch: Batch index vector. [N_sum]
        :return: Predicted regression values. [B]
        """
        X_n = self.pna(X_n, edge_index, edge_attr, snorm)   # [N_sum, hidden_dim]
        X_n = global_add_pool(X_n, batch) # [B, hidden_dim]
        Y_pred = self.mlp(X_n)         # [B, 1]
        return Y_pred.squeeze(dim=1)                  # [B]


