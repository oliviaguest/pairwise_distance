from __future__ import division

import multiprocessing
import scipy.spatial.distance
import numpy as np
import sklearn.datasets
from time import time
from multiprocessing import Pool
from itertools import combinations

N = 100
centers = [[0, 0], [1, 0], [0.5, np.sqrt(0.75)]]
# The SDs:
cluster_std = [0.3, 0.3, 0.3]
# The number of Gaussians:
n_clusters = len(centers)
# The number of points, recall we need double at the top point hence 3/4
# of points are being generated now.
n_samples = int(0.75 * N)
data, labels_true = sklearn.datasets.make_blobs(n_samples=n_samples,\
                    centers=centers, cluster_std=cluster_std)
# Now to generate the extra data points for the top of the triangle:
centers = [[0.5, np.sqrt(0.75)]]
cluster_std = [0.3]
n_clusters = len(centers)
extra, labels_true = sklearn.datasets.make_blobs(n_samples=int(0.25*N),\
                     centers=centers, cluster_std=cluster_std)
# Merge the points together to create the full dataset.
X = np.concatenate((data, extra), axis=0)
t = time()

try:
    Y = scipy.spatial.distance.pdist(X, 'euclidean')
    print Y.sum()
    print '{} s'.format(time() -t)

except MemoryError:
    pass
def calculate_pairwise_distance(a, b):
    return np.linalg.norm(a - b)
t = time()
comb_sum = 0
for comb in combinations(range(X.shape[0]), 2):
    comb_sum += calculate_pairwise_distance(X[comb[0]], X[comb[1]])
print comb_sum
print '{} s'.format(time() -t)
p = Pool()
results = []
# using apply_async
t = time()
# for arg in zip([x+1 for x in r],r[1:]):
for comb in combinations(xrange(X.shape[0]), 2):
    arg = (X[comb[0]].copy(), X[comb[1]].copy())
    results.append(p.apply_async(calculate_pairwise_distance, arg))
# # wait for results
print sum(res.get() for res in results)
print '{} s'.format(time() -t)
