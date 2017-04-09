import dask
import sklearn.datasets
import scipy.spatial.distance

import dask.array as da
import numpy as np

from time import time


# Generate some data:
# for i in range(100): # you can create huge arrays here and won't kill RAM
# because of how cool dask is!
for i in range(100):
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
    try:
        X = da.concatenate([X, da.from_array(np.concatenate((data, extra), axis=0), chunks=(1000,2))], axis=0)
    except NameError:
        X = da.from_array(np.concatenate((data, extra), axis=0), chunks= (1000, 2))

N = X.shape[0]
del data, extra, labels_true

def distance(a, b):
    """ Slow version of ``add`` to simulate work """
    return np.sum(np.sqrt(np.sum((a - b)**2, axis=1)))

# Parallel:
t = time()
pairs = [dask.do(distance)(X[i:], X[:N-i]) for i in xrange(1, N)]
result = dask.do(sum)(pairs)
my_sum = result.compute()
print 'parallel:\t{} s'.format(time() - t)

# Serial:
# Comment this out if you use a high N as it will eat RAM!
# SERIOUSLY, be careful.
t = time()
Y = scipy.spatial.distance.pdist(X, 'euclidean')
print 'serial:\t\t{} s'.format(time() - t)
assert np.round(np.sum(Y)) == np.round(
        my_sum) # There is minor rounding error after 8 decimal places.
print 'sum = {} s'.format(my_sum)
print 'array size:', X.shape, N
