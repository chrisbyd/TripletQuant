import numpy as np
import torch.utils.data as util_data
from torchvision import transforms
import torch
from PIL import Image
from tqdm import tqdm
from utils.distance import eachPQDist
from utils.distance import sign
import torchvision.datasets as dsets
import sys






def euclidean2(x1, x2):
    return np.sum(np.square(x1 - x2), axis=-1)


def euclidean(x1, x2):
    return np.sqrt(euclidean2(x1, x2))



def compute_result(dataloader, net, device):
    bs, clses = [], []
    net.eval()
    with torch.no_grad():
        for img, cls, _ in tqdm(dataloader):
            clses.append(cls)
            bs.append((net(img.to(device))).data.cpu())
    return torch.cat(bs).sign(), torch.cat(clses)

def compute_result2(dataloader, net):
    bs, clses = [], []
    net.eval()
    with torch.no_grad():
        for img, cls, _ in tqdm(dataloader):
            clses.append(cls)
            bs.append((net(img.cuda()).data.cpu()))
    return torch.cat(bs), torch.cat(clses)


def CalcHammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH

def distance(x1, x2=None, pair=True, dist_type="euclidean2", ifsign=False):
    '''
    Param:
        x2: if x2 is None, distance between x1 and x1 will be returned.
        pair: if True, for i, j, x1_i, x2_j will be calculated
              if False, for i, x1_i, x2_i will be calculated, and it requires the dimension of x1 and x2 is same.
        dist_type: distance type, can be euclidean2, normed_euclidean2, inner_product, cosine
    '''
    c = x1 - x2
    a = np.sqrt( np.square(c).sum(-1))
    return a



def CalcTopMap(qB, rB, queryL, retrievalL, topk, distance_metric='hamming', config=None):
    num_query = queryL.shape[0]
    num_gallery = retrievalL.shape[0]
    topkmap = 0
    prec = np.zeros((num_query,num_gallery))
    recall = np.zeros((num_query,num_gallery))
    for iter in tqdm(range(num_query)):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        if distance_metric == 'hamming':
            hamm = CalcHammingDist(qB[iter, :], rB)
        elif distance_metric == 'euclidean':
            hamm = distance(qB[iter, :], rB, pair= True,dist_type="euclidean2")
        else:
            hamm = eachPQDist(qB[iter, :], rB, config['n_book'])
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd).astype(int)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)
        all_sim_num = np.sum(gnd)

        prec_gnd = gnd
        prec_sum = np.cumsum(gnd)
        return_images = np.arange(1,num_gallery+1)
        prec[iter,:] = prec_sum / return_images
        recall[iter,:] = prec_sum / all_sim_num

        assert recall[iter, -1] == 1.0
        assert all_sim_num == prec_sum[-1]

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    index = np.argwhere(recall[:, -1] == 1.0)
    index = index.squeeze()
    prec = prec[index]
    recall = recall[index]
    cum_prec = np.mean(prec, 0)
    cum_recall = np.mean(recall, 0)

    return topkmap, cum_prec , cum_recall