import torch
import torch.nn as nn
from .functions import cross_entropy
__all__ = ['focal_loss', 'cosine', 'vector_length']


def focal_loss(logit: torch.Tensor, gamma, alpha=1, eps=1e-5):
    """
    focal loss: /alpha * (1 - logit)^{gamma} * log^{logit}
    :param logit: logit value 0 ~ 1
    :param gamma:
    :param alpha:
    :param eps: a tiny value prevent log(0)
    :return:
    """
    return -alpha * torch.pow(1 - logit, gamma) * torch.log(logit + eps)


def cosine(hash1: torch.Tensor, hash2: torch.Tensor):
    inter = torch.matmul(hash1, hash2.t())
    length1 = vector_length(hash1, keepdim=True)
    length2 = vector_length(hash2, keepdim=True)
    return torch.div(inter, torch.matmul(length1, length2.t()))


def vector_length(vector: torch.Tensor, keepdim=False):
    if len(vector.shape) > 2:
        vector = vector.unsqueeze(0)
    return torch.sqrt(torch.sum(torch.pow(vector, 2), dim=-1, keepdim=keepdim))

def N_PQ_loss(labels_Simililarity, embedding_x, embedding_q, reg_lambda=0.002):
    reg_anchor = torch.mean(torch.sum(torch.square(embedding_x), 1))
    reg_positive = torch.mean(torch.sum(torch.square(embedding_q), 1))
    l2loss = torch.mul(0.25*reg_lambda, reg_anchor+reg_positive)
    FQ_Similarity = torch.matmul(embedding_x, embedding_q.t())
    #print('???', FQ_Similarity)
    #print('!!!', labels_Simililarity)
    #print('FQ Sim', FQ_Similarity.shape, labels_Simililarity.shape)
    loss = cross_entropy(FQ_Similarity, labels_Simililarity)
    return  loss + l2loss

def raw_PQ_loss(labels_sim, x, q):
    sim = torch.matmul(x, q.t())
    return cross_entropy(sim, labels_sim)

def raw_hash_loss(x, q):
    x2 = torch.square(x)
    q2 = torch.square(q)
    return torch.mean(torch.square(x2-q2))

def CLS_loss(label, logits):
    torch_cross_entropy = torch.nn.CrossEntropyLoss()
    # print('what is label?', label.shape)
    loss = torch.mean(cross_entropy(logits, label))
    return loss

def CLS_bce_loss(label, logits):
    bce_loss = nn.BCEWithLogitsLoss()
    loss = bce_loss(logits, label)
    return loss

def cross_entropy(pred, soft_targets):
    logsoftmax = torch.nn.LogSoftmax(dim=1)
    return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), dim=1))

class NpairLoss(nn.Module):
    """
    The n-pair quantization loss from

    """
    def __init__(self,reg_lambda= 0.002):
        super(NpairLoss, self).__init__()
        self.reg_lambda = reg_lambda

    def forward(self,labels_Simililarity, embedding_x, embedding_q):
        reg_anchor = torch.mean(torch.sum(torch.square(embedding_x), 1))
        reg_positive = torch.mean(torch.sum(torch.square(embedding_q), 1))
        l2loss = torch.mul(0.25* self.reg_lambda, reg_anchor+reg_positive)
        FQ_Similarity = torch.matmul(embedding_x, embedding_q.t())
        #print('???', FQ_Similarity)
        #print('!!!', labels_Simililarity)
        #print('FQ Sim', FQ_Similarity.shape, labels_Simililarity.shape)
        loss = cross_entropy(FQ_Similarity, labels_Simililarity)
        return  loss + l2loss












# class TripletLoss(torch.nn.Module):
#     def __init__(self, bit):
#         super(TripletLoss, self).__init__()

#     def forward(self, u, u1, y):

#         inner_product = 1 - u @ u1.t()
#         s = y @ y.t() > 0
#         count = 0

#         loss1 = 0
#         for row in range(s.shape[0]):
#             # if has positive pairs and negative pairs
#             if s[row].sum() != 0 and (~s[row]).sum() != 0:
#                 count += 1
#                 theta_positive = inner_product[row][s[row] == 1]
#                 theta_negative = inner_product[row][s[row] == 0]
#                 triple = (theta_positive.unsqueeze(1) - theta_negative.unsqueeze(0) - 5).clamp(min=-100,
#                                                                                                              max=50)
#                 loss1 += -(triple - torch.log(1 + torch.exp(triple))).mean()

#         if count != 0:
#             loss1 = loss1 / count
#         else:
#             loss1 = 0

      

#         return loss1 















class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels, m_features, m_labels, self_contrast = True):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

  
        batch_size = features.shape[0]
       
   
        mask = (torch.matmul(labels,m_labels.T) > 0).float().to(device)
          
            
      
        contrast_feature = m_features
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = features
            anchor_count = 1
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
      #  mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        if self_contrast:
            mask = mask * logits_mask
        else:
            mask = mask
        # compute log_prob
        if self_contrast:
            exp_logits = torch.exp(logits) * logits_mask
        else:
            exp_logits = torch.exp(logits) 
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

