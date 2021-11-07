import torch
from . import  common_functions as c_f


def convert_to_weights(indices_tuple, labels, dtype):
    """
    Returns a weight for each batch element, based on
    how many times they appear in indices_tuple.
    """
    weights = torch.zeros(labels.shape[0], device=labels.device)
    weights = c_f.to_dtype(weights, dtype=dtype)
    if (indices_tuple is None) or (all(len(x) == 0 for x in indices_tuple)):
        return weights + 1
    indices, counts = torch.unique(torch.cat(indices_tuple, dim=0), return_counts=True)
    counts = c_f.to_dtype(counts, dtype=dtype) / torch.sum(counts)
    weights[indices] = counts / torch.max(counts)
    return weights


def get_all_pairs_indices(labels, ref_labels=None):
    """
    Given a tensor of labels, this will return 4 tensors.
    The first 2 tensors are the indices which form all positive pairs
    The second 2 tensors are the indices which form all negative pairs
    """
    if ref_labels is None:
        ref_labels = labels
    
    matches = labels @ labels.t()
    matches = matches > 0
   
    diffs = matches <= 0
    if ref_labels is labels:
        matches.fill_diagonal_(0)
    a1_idx, p_idx = torch.where(matches)
    a2_idx, n_idx = torch.where(diffs)
    return a1_idx, p_idx, a2_idx, n_idx

class FastAPLoss(object):
    def __init__(self, num_bins=10):
        super().__init__()
        
        self.num_bins = int(num_bins)
        self.num_edges = self.num_bins + 1
        

    """
    Adapted from https://github.com/kunhe/FastAP-metric-learning
    """

    def compute_loss(self, embeddings,descriptor, labels, indices_tuple=None):
        dtype, device = embeddings.dtype, embeddings.device
        miner_weights = convert_to_weights(indices_tuple, labels, dtype=dtype)
        N = labels.size(0)
        a1_idx, p_idx, a2_idx, n_idx = get_all_pairs_indices(labels)
        I_pos = torch.zeros(N, N, dtype=dtype, device=device)
        I_neg = torch.zeros(N, N, dtype=dtype, device=device)
        I_pos[a1_idx, p_idx] = 1
        I_neg[a2_idx, n_idx] = 1
        N_pos = torch.sum(I_pos, dim=1)
        safe_N = N_pos > 0
        if torch.sum(safe_N) == 0:
            return self.zero_losses()
     #   dist_mat = self.distance(embeddings)
        dist_mat = 2 - 2 * torch.mm(embeddings, descriptor.t())

        histogram_max = 2 ** 2
        histogram_delta = histogram_max / self.num_bins
        mid_points = torch.linspace(
            0.0, histogram_max, steps=self.num_edges, device=device, dtype=dtype
        ).view(-1, 1, 1)
        pulse = torch.nn.functional.relu(
            1 - torch.abs(dist_mat - mid_points) / histogram_delta
        )
        pos_hist = torch.t(torch.sum(pulse * I_pos, dim=2))
        neg_hist = torch.t(torch.sum(pulse * I_neg, dim=2))

        total_pos_hist = torch.cumsum(pos_hist, dim=1)
        total_hist = torch.cumsum(pos_hist + neg_hist, dim=1)

        h_pos_product = pos_hist * total_pos_hist
        safe_H = (h_pos_product > 0) & (total_hist > 0)
        if torch.sum(safe_H) > 0:
          
            FastAP = torch.zeros_like(pos_hist, device=device)
            FastAP[safe_H] = h_pos_product[safe_H] / total_hist[safe_H]
            FastAP = torch.sum(FastAP, dim=1)
            FastAP = FastAP[safe_N] / N_pos[safe_N]
            FastAP = (1 - FastAP) * miner_weights[safe_N]
            FastAP = torch.mean(FastAP)
            return FastAP
        return self.zero_losses()
