from __future__ import division

import multiprocessing
import scipy.spatial.distance
import numpy as np
import sklearn.datasets
from time import time
import multiprocessing as mp
from itertools import combinations
# Generate some data ###########################################################
N = 100
centers = [[0, 0], [1, 0], [0.5, np.sqrt(0.75)]]
# The SDs:
cluster_std = [0.3, 0.3, 0.3]
n_clusters = len(centers)
n_samples = int(0.75 * N)
data, labels_true = sklearn.datasets.make_blobs(n_samples=n_samples,\
                    centers=centers, cluster_std=cluster_std)
centers = [[0.5, np.sqrt(0.75)]]
cluster_std = [0.3]
n_clusters = len(centers)
extra, labels_true = sklearn.datasets.make_blobs(n_samples=int(0.25*N),\
                     centers=centers, cluster_std=cluster_std)
X = np.concatenate((data, extra), axis=0)
# X = np.asarray([[i] for i in range(5)])

################################################################################
# Now do it the scipy way ######################################################
try:
    t = time()
    Y = scipy.spatial.distance.pdist(X, 'euclidean')
    print Y.sum()
    print '{} s'.format(time() -t)
except MemoryError:
    pass
################################################################################
# Now the way I want to but without multiprocessing ############################
def calculate_pairwise_distance(a, b):
    return np.linalg.norm(a - b)
# t = time()
# comb_sum = 0
# for comb in combinations(range(X.shape[0]), 2):
#     comb_sum += calculate_pairwise_distance(X[comb[0]], X[comb[1]])
# print comb_sum
# print '{} s'.format(time() -t)
################################################################################
# And finally the way I want to with multiprocessing ###########################
p = mp.Pool(20)
results = []
t = time()
for comb in combinations(xrange(X.shape[0]), 2):
    arg = (X[comb[0]].copy(), X[comb[1]].copy())
    results.append(p.apply_async(calculate_pairwise_distance, arg))
# print results.get()
print sum(res.get() for res in results)
print '{} s'.format(time() -t)
################################################################################
# A different way with multiprocessing that does not work! #####################
# Traceback (most recent call last):
#   File "pairwise_distance.py", line 63, in <module>
#     results.append(pool.map(calculate_pairwise_distance, X[comb[0]], X[comb[1]]))
#   File "/home/olivia/anaconda2/lib/python2.7/multiprocessing/pool.py", line 251, in map
#     return self.map_async(func, iterable, chunksize).get()
#   File "/home/olivia/anaconda2/lib/python2.7/multiprocessing/pool.py", line 314, in map_async
#     result = MapResult(self._cache, chunksize, len(iterable), callback)
#   File "/home/olivia/anaconda2/lib/python2.7/multiprocessing/pool.py", line 594, in __init__
#     if chunksize <= 0:
# ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
# t = time()
# pool = mp.Pool(processes=mp.cpu_count())
# for comb in combinations(xrange(X.shape[0]), 2):
#     results.append(pool.map(calculate_pairwise_distance, X[comb[0]], X[comb[1]]))
# pool.close()
# print sum(res.get() for res in results)
# print '{} s'.format(time() -t)
################################################################################
t = time()
distance = 0
N = X.shape[0]
for i in xrange(N):
    if i:
        #
        # print X[i:]
        # print X[:N-i]
        # print i, N-i
        distance += np.linalg.norm(X[i:] - X[:N-i])
print distance
print '{} s'.format(time() -t)
