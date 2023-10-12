import torch
import numpy as np
import matplotlib.pyplot as plt
from torchmetrics.classification import BinaryAUROC

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


def dataset_statistics(dataset, visualization=False):
    # graph size distribution
    size = []
    for data in dataset:
        size.append(int(data.num_nodes))
    size = np.array(size)
    print('Mean (std) of graph size is %.3f+-%.3f'%(size.mean(), size.std()))
    print('Max/min graph size is %d, %d' % (size.max(), size.min()))
    if visualization:
        fig, axs = plt.subplots(2)
        axs[0].hist(size, bins=size.max()-size.min())
        axs[0].set_title("Graph size distribution")

    # eigenvalues distribution
    eig_mults = []
    for data in dataset:
        Lambda = data.Lambda[0, :data.num_nodes]
        rounded_Lambda = torch.unique(around(Lambda))
        eig_mults.append(Lambda.size(-1) - rounded_Lambda.size(0) + 1)

    eig_mults = np.array(eig_mults)
    print("Percentage of graphs that has multiplicity: %.3f" % (np.mean(eig_mults > 1) * 100))
    print("Multiplicity mean(std) = %.3f+-%.3f" % (eig_mults.mean(), eig_mults.std()))
    if visualization:
        axs[1].hist(eig_mults, bins=eig_mults.max()-eig_mults.min())
        axs[1].set_title("Eigenvalues multiplicity distribution")
        plt.show()




def classification_analysis(y_pred, y_label, num_nodes=None):
    # analysis the classification behavior
    # ground truth analysis
    num_p, num_n = (y_label.int() == 1).float().sum().item(), (y_label.int() == 0).float().sum().item()
    print("Ground truth: %.3f%% are positive, %.3f%% are negative"%(100*num_p / (num_p+num_n), 100*num_n/(num_p+num_n)))

    # prediction analysis
    pred = (y_pred > 0.5).float()
    num_pred_p, num_pred_n = (pred.int() == 1).float().sum().item(), (pred.int() == 0).float().sum().item()
    TP = (pred * y_label).sum()
    TN = (torch.abs(pred - 1) * torch.abs(y_label - 1)).sum()
    FP = num_pred_p - TP
    FN = num_pred_n - TN
    TP, TN, FP, FN = TP / num_p, TN / num_n, FP / num_n, FN / num_p
    print("Rate of TP | TN | FP | FN: %.3f | %.3f | %.3f | %.3f" % (TP, TN, FP, FN))


    # logits analysis
    logits_n = y_pred[torch.where(y_label == 0.0)].mean(), y_pred[torch.where(y_label == 0.0)].std()
    logits_p = y_pred[torch.where(y_label == 1.0)].mean(), y_pred[torch.where(y_label == 1.0)].std()
    print("Probability for positive samples: %.3f+-%.3f" % logits_p)
    print("Probability for negative samples: %.3f+-%.3f" % logits_n)


    # analysis w.r.t #nodes
    if num_nodes is not None:
        sort, index = torch.sort(num_nodes)
        y_pred_all = y_pred[index]
        y_label_all = y_label[index]
        # each 0.1 quantile:
        n = 10
        for i in range(n):
            print("----- Number of nodes ranging from %d to %d -----" % (sort[int(i/n*len(sort))],
                                                                         sort[int((i+1)/n*len(sort))-1]))
            y_pred = y_pred_all[int(i/n*len(sort)): int((i+1)/n*len(sort))]
            y_label = y_label_all[int(i/n*len(sort)): int((i+1)/n*len(sort))]
            pred = (y_pred > 0.5).float()
            num_pred_p, num_pred_n = (pred.int() == 1).float().sum().item(), (pred.int() == 0).float().sum().item()
            num_p, num_n = (y_label.int() == 1).float().sum().item(), (y_label.int() == 0).float().sum().item()
            print("Ground truth: %.3f%% are positive, %.3f%% are negative" % (
            100 * num_p / (num_p + num_n), 100 * num_n / (num_p + num_n)))
            TP = (pred * y_label).sum()
            TN = (torch.abs(pred - 1) * torch.abs(y_label - 1)).sum()
            FP = num_pred_p - TP
            FN = num_pred_n - TN
            TP, TN, FP, FN = TP / num_p, TN / num_n, FP / num_n, FN / num_p
            auc = BinaryAUROC()(y_pred, y_label)
            print("Rate of TP | TN | FP | FN: %.3f | %.3f | %.3f | %.3f" % (TP, TN, FP, FN))
            print("AUC: %.3f" % auc)

            print("---------------")
