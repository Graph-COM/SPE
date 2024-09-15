import logging
import os
import random
import uuid
from typing import TextIO, Optional, List, Dict, Any

import numpy as np
import torch
import wandb
from omegaconf import OmegaConf
from torch import optim, nn
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau

# dataset and dataloader
from torch_geometric.data import Data, Batch
from torch_geometric.utils import degree
# from torch_geometric.datasets import ZINC
from dataset import ZINC # customized ZINC with processed file re-naming support
# from torch_geometric.loader import DataLoader
from src.data_utils.dataloder import DataLoader # customized dataloder to handle BasisNet features
# from src.data_utils.dataloder import SameSizeDataLoader # DO NOT USE NOW: currently encounter instability of training
from torch_geometric.utils import get_laplacian, to_dense_adj

# models
from root import root
from src.mlp import MLP
from src.model import Model, construct_model
from src.schema import Schema



from src.utils import eigenvalue_multiplicity, get_projections

class Trainer:
    cfg: Schema
    model: Model
    train_loader: DataLoader
    val_loader: DataLoader
    optimizer: optim.Adam
    criterion: nn.L1Loss
    metric: nn.L1Loss
    logger: logging.Logger
    val_writer: TextIO
    curr_epoch: int
    curr_batch: int

    def __init__(self, cfg: Schema, gpu_id: Optional[int]) -> None:
        set_seed(cfg.seed)

        # Initialize configuration
        self.cfg = cfg
        cfg.out_dirpath = root(cfg.out_dirpath)

        # Construct model
#        base_model = GINEBaseModel(
#            cfg.n_base_layers, cfg.n_edge_types, cfg.node_emb_dims, cfg.base_hidden_dims, self.create_mlp
#        )
        # self.model = self.construct_model(cfg, base_model) # final model = pe_method + base_model
        # self.model = construct_model(cfg, self.create_mlp)
        # sanity check
#        self.model = Model(
#            cfg.n_node_types, cfg.node_emb_dims,
#            positional_encoding=NoPE(cfg.pe_dims),
#            base_model=base_model
#        )

        # self.model.to("cpu" if gpu_id is None else f"cuda:{gpu_id}")

        # Construct data loaders
        ## dataset preprocessing (before saved in disk) and loading
        processed_suffix = '_pe' + str(cfg.pe_dims) if cfg.pe_method != 'none' else ''
        transform = self.get_projs if cfg.pe_method == 'basis_inv' else self.get_snorm
        pre_transform = self.pre_transform if cfg.pe_method != 'none' else None
        train_dataset = ZINC(root("data/zinc"), subset=cfg.use_subset, split="train", pre_transform=pre_transform,
                             transform=transform, processed_suffix=processed_suffix)
        val_dataset = ZINC(root("data/zinc"), subset=cfg.use_subset, split="val", pre_transform=pre_transform,
                           transform=transform, processed_suffix=processed_suffix)
        test_dataset = ZINC(root("data/zinc"), subset=cfg.use_subset, split="test", pre_transform=pre_transform,
                           transform=transform, processed_suffix=processed_suffix)


        # self.train_loader = DataLoader(train_dataset[:cfg.subset_size], batch_size=cfg.train_batch_size, shuffle=True)
        self.train_loader = DataLoader(train_dataset, batch_size=cfg.train_batch_size, shuffle=True, num_workers=4)
        self.val_loader = DataLoader(val_dataset, batch_size=cfg.val_batch_size, shuffle=False, num_workers=0)
        self.test_loader = DataLoader(test_dataset, batch_size=cfg.val_batch_size, shuffle=False, num_workers=0)


        # construct model after loading dataset
        kwargs = {"uniq_mults": [1, 2, 3, 4, 5]} if cfg.pe_method == 'basis_inv' else {} # only works for pe-dim=8
        # kwargs = {"uniq_mults": [i for i in range(1, 10)]} if cfg.pe_method == 'basis_inv' else {} # only works for pe-dim=32
        kwargs["deg"] = self.get_degree(train_dataset) if cfg.base_model == 'pna' else None
        kwargs["device"] = f"cuda:{gpu_id}"
        kwargs["residual"] = cfg.residual
        kwargs["bn"] = cfg.batch_norm
        kwargs["sn"] = cfg.graph_norm
        kwargs["feature_type"] = "discrete"
        # self.model = construct_model(cfg, self.create_mlp, **kwargs)
        self.model = construct_model(cfg, self.create_mlp, **kwargs)
        self.model.to("cpu" if gpu_id is None else f"cuda:{gpu_id}")

        # Construct auxiliary training objects
        param_groups = self.get_param_groups()
        # self.optimizer = optim.SGD(param_groups, lr=cfg.lr, momentum=cfg.momentum, nesterov=cfg.nesterov)
        # I generally find Adam is better than SGD
        self.optimizer = optim.Adam(param_groups, lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.scheduler = LambdaLR(self.optimizer, self.lr_lambda)
        # self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=25)
        self.criterion = nn.L1Loss(reduction="mean")
        self.metric = nn.L1Loss(reduction="sum")

        # Set up logger and writer
        name = str(uuid.uuid4())
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        os.makedirs(cfg.out_dirpath, exist_ok=True)
        handler = logging.FileHandler(os.path.join(cfg.out_dirpath, "train_logs.txt"))
        handler.setLevel(logging.INFO)

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", "%Y/%m/%d %H:%M:%S")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        self.val_writer = open(os.path.join(cfg.out_dirpath, "evaluate_logs.txt"), "a")

        # Set up WandB
        wandb.login(key="") # use your own WanbB key
        cfg.__dict__['num_params'] = sum(param.numel() for param in self.model.parameters())
        wandb.init(dir=root("."), project="SPE", config=cfg.__dict__)
        #wandb.run.log_code(".")

        # Miscellaneous
        self.curr_epoch = 1
        self.curr_batch = 1

    def train(self) -> None:
        self.logger.info("Configuration:\n" + OmegaConf.to_yaml(self.cfg))
        self.logger.info(f"Total parameters: {sum(param.numel() for param in self.model.parameters())}")
        self.logger.info(f"Total training steps: {self.n_total_steps}")
        self.logger.info(
            "Optimizer groups:\n" + "\n".join(group["name"] for group in self.optimizer.param_groups) + "\n")

        best_val_loss, best_test_loss = 999.0, 999.0

        for self.curr_epoch in range(1, self.cfg.n_epochs + 1):
            train_loss = self.train_epoch()
            val_loss = self.evaluate(self.val_loader)
            test_loss = self.evaluate(self.test_loader)
            # self.scheduler.step(eval_loss)
            # lr = self.scheduler.get_last_lr()[0]
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            wandb.log({'train_loss': train_loss, 'eval_loss': val_loss, 'test_loss': test_loss, 'lr': lr})
            if val_loss < best_val_loss:
                best_val_loss, best_test_loss = val_loss, test_loss
                wandb.run.summary["best_val_loss"] = best_val_loss
                wandb.run.summary["best_test_loss"] = best_test_loss
                wandb.run.summary["best_training_loss"] = train_loss

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0
        for self.curr_batch, batch in enumerate(self.train_loader, 1):
            total_loss += self.train_batch(batch)

        return total_loss / len(self.train_loader.dataset)


    def train_batch(self, batch: Batch) -> float:
        batch.to(device(self.model))
        self.optimizer.zero_grad()

        y_pred = self.model(batch)               # [B]
        loss = self.criterion(y_pred, batch.y)   # [1]
        loss.backward()
        self.optimizer.step()

        loss = loss.item()
        # lr = self.scheduler.get_last_lr()[0]
        # self.logger.info("Training... Epoch: {}, batch: {}, loss: {:.6f}, lr: {:.6e}"
        #        .format(self.curr_epoch, self.curr_batch, loss, lr))
        # wandb.log({"train_loss": loss, "lr": lr})

        self.scheduler.step()

        return loss * batch.y.size(0)
        # return loss

    def evaluate(self, eval_loader: DataLoader) -> float:
        self.model.eval()

        total_loss = 0.0
        for self.curr_batch, batch in enumerate(eval_loader, 1):
            total_loss += self.evaluate_batch(batch)
            self.logger.info(f"Evaluating... Epoch: {self.curr_epoch}, batch: {self.curr_batch}")
        total_loss /= len(eval_loader.dataset)

        self.val_writer.write(f"Epoch: {self.curr_epoch}\t Loss: {total_loss}\n")
        self.val_writer.flush()
        # wandb.log({"val_loss": total_loss})
        return total_loss

    def evaluate_batch(self, batch: Batch) -> float:
        batch.to(device(self.model))
        with torch.no_grad():
            y_pred = self.model(batch)
        return self.metric(y_pred, batch.y).item()

    @property
    def n_total_steps(self) -> int:
        return len(self.train_loader) * self.cfg.n_epochs

    def create_mlp(self, in_dims: int, out_dims: int) -> MLP:
        return MLP(
            self.cfg.n_mlp_layers, in_dims, self.cfg.mlp_hidden_dims, out_dims, self.cfg.mlp_use_bn,
            self.cfg.mlp_activation, self.cfg.mlp_dropout_prob
         )

    def create_mlp_ln(self, in_dims: int, out_dims: int) -> MLP:
        return MLP(
            self.cfg.n_mlp_layers, in_dims, self.cfg.mlp_hidden_dims, out_dims, self.cfg.mlp_use_ln,
            self.cfg.mlp_activation, self.cfg.mlp_dropout_prob, norm_type="layer"
        )

    def get_projs(self, instance: Data) -> Data:
        # get projection matrices on the fly
        projs, mults = get_projections(eigvals=instance.Lambda, eigvecs=instance.V)
        instance.update({"P": projs, "mults": mults})
        return instance

    def get_snorm(self, instance: Data) -> Data:
        # get the graph normalization for nodes on the fly
        size = instance.num_nodes
        snorm = torch.FloatTensor(size, 1).fill_(1./float(size)).sqrt()
        instance.update({"snorm": snorm})
        return instance

    def pre_transform(self, instance: Data) -> Data:
        # get spectrum
        n = instance.num_nodes
        L_edge_index, L_values = get_laplacian(instance.edge_index, normalization="sym", num_nodes=n)   # [2, X], [X]
        L = to_dense_adj(L_edge_index, edge_attr=L_values, max_num_nodes=n).squeeze(dim=0)              # [N, N]

        Lambda = torch.zeros(1, self.cfg.pe_dims)   # [1, D_pe]
        V = torch.zeros(n, self.cfg.pe_dims)        # [N, D_pe]

        #d = min(n - 1, self.cfg.pe_dims)   # number of eigen-pairs to use (then we zero-pad up to D_pe)
        d = min(n, self.cfg.pe_dims)   # number of eigen-pairs to use (then we zero-pad up to D_pe)
        eigenvalues, eigenvectors = torch.linalg.eigh(L)   # [N], [N, N]
        #Lambda[0, :d] = eigenvalues[1:d + 1]
        #V[:, :d] = eigenvectors[:, 1:d + 1]
        Lambda[0, :d] = eigenvalues[0:d]
        V[:, :d] = eigenvectors[:, 0:d]

        instance.update({"Lambda": Lambda, "V": V})

        return instance

    def get_param_groups(self) -> List[Dict[str, Any]]:
        return [{
            "name": name,
            "params": [param],
            "weight_decay": 0.0 if "bias" in name else self.cfg.weight_decay
        } for name, param in self.model.named_parameters()]

    def lr_lambda(self, curr_step: int) -> float:
        """
        Based on https://github.com/huggingface/transformers/blob/v4.28.1/src/transformers/optimization.py#L79
        """
        if curr_step < self.cfg.n_warmup_steps:
            return curr_step / max(1, self.cfg.n_warmup_steps)
        else:
            return max(0.0, (self.n_total_steps - curr_step) / max(1, self.n_total_steps - self.cfg.n_warmup_steps))

    def get_degree(self, train_dataset):
        # reference: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/pna.py
        # Compute the maximum in-degree in the training data.
        max_degree = -1
        for data in train_dataset:
            d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            max_degree = max(max_degree, int(d.max()))

        # Compute the in-degree histogram tensor
        deg = torch.zeros(max_degree + 1, dtype=torch.long)
        for data in train_dataset:
            d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            deg += torch.bincount(d, minlength=deg.numel())
        return deg


def set_seed(seed: int) -> None:
    """
    Based on https://github.com/huggingface/transformers/blob/v4.28.1/src/transformers/trainer_utils.py#L83
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device
