# On the Expressivity of Stable Positional Encodings for Graph Neural Networks

## About

This is the official code for the paper "On the Expressivity of Stable Positional Encodings for Graph Neural Networks". 

Feel free to contact yinan8114@gmail.com if there is any question.

![model](model.png)

## Introduction

In this work, we present SPE, a Laplacian-based graph positional encodings that are provably stable and expressive. The key insight is to perform a **soft and learnable** ``partition" of eigensubspaces in an **eigenvalue dependent** way, hereby achieving both stability (from the soft partition) and expressivity (from dependency on both eigenvalues and eigenvectors). 


Our SPE method processes eigenvectors $V\in\mathbb{R}^{n\times d}$ and eigenvalues $\lambda\in\mathbb{R}^d$ into node positional encodings as follows:
$$\text{SPE}(V, \lambda)=\rho(V\text{diag}\{\phi_1(\lambda)\}V^{T}, V\text{diag}\{\phi_2(\lambda)\}V^{T}, ..., V\text{diag}\{\phi_m(\lambda)\}V^{T}),$$
where $\rho:\mathbb{R}^{n\times n\times m}\to\mathbb{R}^{n\times p}$ and $\phi_i:\mathbb{R}^{d}\to\mathbb{R}^d$ are permutational equivariant functions w.r.t. $n\times n$ and $d$ axes respectively.

## Code usage

### Requirements

See requirements.txt for necessary python environment.

### Reproduce experiments

To reproduce experiments on ZINC, cd to ./zinc and run
```
python runner.py --config_dirpath ../configs/zinc --config_name SPE_gine_gin_mlp_pe37.yaml --seed 0
```


To reproduce experiments on Alchemy, cd to ./alchemy and run
```
python --config_dirpath ../configs/alchemy --config_name SPE_gine_gin_mlp_pe12.yaml --seed 0
```

To reproduce experiments on DrugOOD, cd to ./drugood and run
```
python --config_dirpath ../configs/assay --config_name SPE_gine_gin_mlp_pe32_zeropsi.yaml --dataset assay --seed 0
python --config_dirpath ../configs/scaffold --config_name SPE_gine_gin_mlp_pe32_standard_dropout.yaml --dataset scaffold --seed 0
python --config_dirpath ../configs/scaffold --config_name SPE_gine_gin_mlp_pe32_standard_dropout.yaml --dataset size --seed 0
```

To reproduce substructures counting, cd to ./count and run
```
bash run.sh
```
