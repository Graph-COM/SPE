import torch
import numpy as np

def mask2d_sum_pooling(x, mask):
    #  x: [B, N, N, d], mask: [B, N, N, 1]
    return (x * mask).sum(dim=1)


def mask2d_diag_offdiag_maxpool(x, mask):
    raise Exception('Not implemented mask2d_diag_offdiag_maxpool yet.')

def mask2d_diag_offdiag_meanpool(x, mask):
    #  x: [B, N, N, d], mask: [B, N, N, 1]
    N = mask.size(1)
    mean_diag = torch.diagonal(x, dim1=1, dim2=2).transpose(-1, -2) # [B, N, d]
    mean_offdiag = (torch.sum(x * mask, dim=1) + torch.sum(x * mask, dim=2) - 2 * mean_diag) / (2 * N - 2)
    return torch.cat((mean_diag, mean_offdiag), dim=-1)  # [B, N, 2*d]


def eigenvalue_multiplicity(dataset):
    percent_graphs = 0.0
    percent_nodes = 0.0
    num_nodes = 0
    for data in dataset:
        Lambda = data.Lambda
        Lambda = np.array(Lambda.view(-1))
        flag_graph = 0
        flag_node = 0
        num_nodes += len(Lambda)
        for x in Lambda:
            flag = np.abs(Lambda - x) < 1e-4
            if np.sum(flag.astype(int)) >= 2:
                flag_graph = 1
                flag_node += 1
        if flag_graph == 1:
            percent_graphs += 1
            percent_nodes += flag_node
    print('Percent of graphs that have multiplicity eigenvalues: %.3f%%'%(100*percent_graphs / len(dataset)))
    print('Percent of nodes that have multiplicity eigenvalues: %.3f%%'%(100*percent_nodes / num_nodes))


def around(x, decimals=5):
    """ round to a number of decimal places """
    return torch.round(x * 10**decimals) / (10**decimals)


def get_projections(eigvals, eigvecs):
    N = eigvecs.size(0)
    pe_dim = np.min([N, eigvals.size(-1)])
    # eigvals, eigvecs = eigvals[:, :N], eigvecs[:, :N]
    rounded_vals = around(eigvals, decimals=5)
    # get rid of the padding zeros
    rounded_vals = rounded_vals[0, :pe_dim]
    uniq_vals, inv_inds, counts = rounded_vals.unique(return_inverse=True, return_counts=True)
    uniq_mults = counts.unique()
    #print('Unique vals', uniq_vals.shape)
    #print('Unique multiplicities', uniq_mults)
    #print('Vals', rounded_vals)
    #print('Multiplicities', counts)
    #print('prop vecs in higher mult', (counts>1).sum()/counts.shape[0])
    # print('prop vecs in higher mult', counts[counts>1].sum()/N)
    sections = torch.cumsum(counts, 0)
    eigenspaces = torch.tensor_split(eigvecs, sections.cpu(), dim=1)[:-1]
    projectors = [V @ V.T for V in eigenspaces]
    projectors = [P.reshape(1,1,N,N) for P in projectors]
    # NUM_EIGENSPACES = len(projectors)
    # print('Num eigenspaces:', NUM_EIGENSPACES)

    same_size_projs = {mult.item(): [] for mult in uniq_mults}
    for i in range(len(projectors)):
        mult = counts[i].item()
        same_size_projs[mult].append(projectors[i])
    for mult, projs in same_size_projs.items():
        same_size_projs[mult] = torch.cat(projs, dim=0)

    # sanity check
    #    temp = 0
    #    for key in same_size_projs.keys():
    #        temp += same_size_projs[key].size(0) * key
    #    assert temp == pe_dim

    return same_size_projs, uniq_mults
