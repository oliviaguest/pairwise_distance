from __future__ import division

import multiprocessing
import sklearn.datasets

import numpy as np
import multiprocessing as mp

from time import time
from itertools import combinations


def calculate_pairwise_distance_sum(t):
    return np.sum(np.sqrt(np.sum((t[0] - t[1])**2, axis=1)))


def calculate_pairwise_distance_tuple(t):
    return np.sqrt(np.sum((t[0] - t[1])**2))


if __name__ == '__main__':
    # Generate some data #####################################################
    N = 1000
    centers = [[0, 0], [1, 0], [0.5, np.sqrt(0.75)]]
    cluster_std = [0.3, 0.3, 0.3]
    n_clusters = len(centers)
    n_samples = int(0.75 * N)
    data, labels_true = sklearn.datasets.make_blobs(n_samples=n_samples,
                                                    centers=centers, cluster_std=cluster_std)
    centers = [[0.5, np.sqrt(0.75)]]
    cluster_std = [0.3]
    n_clusters = len(centers)
    extra, labels_true = sklearn.datasets.make_blobs(n_samples=int(0.25 * N),
                                                     centers=centers, cluster_std=cluster_std)
    X = np.concatenate((data, extra), axis=0)
    N = X.shape[0]
    # weights = np.random.random_sample((N,)) * 100

    ##########################################################################
    # Brad's idea:
    t = time()
    pool = mp.Pool(4)
    results = []
    pairs = [(X[i:], X[:N - i]) for i in xrange(0, N)]
    results = pool.map(calculate_pairwise_distance_sum, pairs, chunksize=1)
    pool.close()
    print sum(results)
    print '{} s'.format(time() - t)

    ##########################################################################
    # My idea:
    t = time()
    p = mp.Pool(12)
    results = []
    combs = [(X[comb[0]], X[comb[1]])
             for comb in combinations(xrange(X.shape[0]), 2)]
    results = p.map(calculate_pairwise_distance_tuple, combs, chunksize=100)
    print sum(results)
    print '{} s'.format(time() - t)
