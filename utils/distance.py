import numpy as np
import sys


def sign(x):
    s = np.sign(x)
    tmp = s[s == 0]
    s[s==0] = np.random.choice([-1, 1], tmp.shape)
    return s

def norm(x, keepdims=False):
    '''
    Param:
        x: matrix of shape (n1, n2, ..., nk)
        keepdims: Whether keep dims or not
    Return: norm of matrix of shape (n1, n2, ..., n_{k-1})
    '''
    return np.sqrt(np.sum(np.square(x), axis=-1, keepdims=keepdims))


def normed(x):
    '''
    Param: matrix of shape (n1, n2, ..., nk)
    Return: normed matrix
    '''
    return x / (1e-20 + norm(x, keepdims=True))


def euclidean2(x1, x2):
    return np.sum(np.square(x1 - x2), axis=-1)


def euclidean(x1, x2):
    return np.sqrt(euclidean2(x1, x2))


def averaged_euclidean2(x1, x2):
    return np.mean(np.square(x1 - x2), axis=-1)


def averaged_euclidean(x1, x2):
    return np.sqrt(averaged_euclidean2(x1, x2))


def normed_euclidean2(x1, x2):
    return euclidean2(normed(x1), normed(x2))


def inner_product(x1, x2, pair=False):
    if pair:
        return - np.inner(x1, x2)
    else:
        return - np.sum(x1 * x2, axis=-1)


def cosine(x1, x2):
    return (1 + inner_product(normed(x1), normed(x2))) / 2

"""
p: [KL]
g: [B, KL]
K is the n_book l is the code len in each book
"""

def eachPQDist(q, g, n_book):
    q = np.expand_dims(q, 0)
    x = np.split(q, n_book, 1)
    y = np.split(g, n_book, 1)
    res = np.zeros((g.shape[0]))
    for n in range(n_book):
        queryCode = x[n] # [L]
        galleryCode = y[n] # [G, L]
        c = queryCode - galleryCode
        dist = np.sqrt( np.square(c).sum(-1))
        #dist = np.squeeze(1 - galleryCode @ queryCode.T, 1)
        #print('dist', dist)
        res += dist 
    return res

def pq_dist(q, g, n_book):
    query_size = q.shape[0]
    gallery_size = g.shape[0]
    dist = np.zeros((query_size, gallery_size))
    for a_q in range(query_size):
        dist[a_q, :] = eachPQDist(q[a_q, :], g, n_book)
    return dist

def distance(x1, x2=None, pair=True, dist_type="euclidean2", ifsign=False, config=None):
    '''
    Param:
        x2: if x2 is None, distance between x1 and x1 will be returned.
        pair: if True, for i, j, x1_i, x2_j will be calculated
              if False, for i, x1_i, x2_i will be calculated, and it requires the dimension of x1 and x2 is same.
        dist_type: distance type, can be euclidean2, normed_euclidean2, inner_product, cosine
    '''
    if x2 is None:
        x2 = x1
    if ifsign:
        x1 = sign(x1)
        x2 = sign(x2)
    if dist_type == 'inner_product':
        return inner_product(x1, x2, pair)
    if dist_type == 'pq':
        n_book = config['n_book']
        return pq_dist(x1, x2, n_book)
    if pair:
        x1 = np.expand_dims(x1, 1)
        x2 = np.expand_dims(x2, 0)
    return getattr(sys.modules[__name__], dist_type)(x1, x2)


if __name__ == "__main__":
    def myAssert(x1, x2):
        assert np.mean(x1 - x2) < 1e-8


    x1 = 2 * np.array([[1, 1, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]])
    x2 = 3 * np.eye(3)
    print(x1)
    print(x2)
    myAssert(distance(x1, x2, pair=True, dist_type="euclidean2"),
             np.array([[9., 9., 9.],
                       [5., 5., 17.],
                       [5., 17., 5.],
                       [17., 5., 5.]]))
    myAssert(distance(x1, x2, pair=True, dist_type="normed_euclidean2"),
             np.array([[0.84529946, 0.84529946, 0.84529946],
                       [0.58578644, 0.58578644, 2.],
                       [0.58578644, 2., 0.58578644],
                       [2., 0.58578644, 0.58578644]]))
    assert distance(x1, x2, pair=True, dist_type="cosine").shape == (4, 3)
    assert distance(x1, x2, pair=True, dist_type="inner_product").shape == (4, 3)

    assert np.all(distance(x1, x1[::-1], pair=False, dist_type="euclidean2") == np.array([4, 8, 8, 4]))
    myAssert(distance(x1, x1[::-1], pair=False, dist_type="normed_euclidean2"),
             np.array([0.36700684, 1., 1., 0.36700684]))
    myAssert(distance(x1, x1[::-1], pair=False, dist_type="cosine"), np.array([0.09175171, 0.25, 0.25, 0.09175171]))
    assert np.all(distance(x1, x1[::-1], pair=False, dist_type="inner_product") == np.array([-8, -4, -4, -8]))

    x3 = np.array([[1,-1,1],[1,1,1],[1,1,-1],[-1,1,-1]])
    x4 = np.array([[1,1,1,],[1,-1,1],[-1,-1,1],[1,-1,1],[-1,-1,1]])
    dis = distance(x3, x4, pair=True, dist_type="inner_product")
    print(dis)

