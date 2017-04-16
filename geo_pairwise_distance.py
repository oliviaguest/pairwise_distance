from __future__ import division

import sklearn.datasets
import scipy.spatial.distance

import multiprocessing as mp
import numpy as np

from time import time
from geopy.distance import great_circle
from scipy.spatial.distance import pdist

##############################
# Code: Olivia Guest         #
# Algorithm: Bradley C. Love #
##############################



def batch_pdist(data_slice):
    # Each data_slice has tuples consisting of two points that we need to
    # find the great circle distance between and their weight:
    partial_sum = 0
    for X, Y, weights in data_slice:
        dist = np.array([])
        zipped = zip(X, Y)
        for x, y in zipped:
            dist = np.append(dist, great_circle(x, y).km)
        partial_sum += np.sum(weights * dist )
    return partial_sum
    # return 10

def mean_pairwise_distance(X, weights=None, n_jobs=None, axis=0):
    """Function that returns the sum and mean of the pairwise distances of an 2D
    array X.

    Required arguments:
    X       --  2D array of points.

    Optional arguments:
    weights -- 1D array of counts or weights per point in X (default: 1s).
    n_jobs  -- Number of cores to use for calculation (default: all).
    axis    -- The axis of X corresponding to data elements (default: 0).
    """
    N = X.shape[axis]
    if weights is None:
        weights = np.ones((N,))
    if n_jobs is None:
        n_jobs = min(mp.cpu_count(),N)
    # Get the pairs and their weights to calculate the distances without
    # needing the whole of X, split it into roughly equal sub-arrays per cpu:
    pairs_split = np.array_split([(X[i:], X[:N - i], weights[i:] * weights[:N - i])
                                  for i in xrange(1, N)],
                                 n_jobs, axis=axis)

    # Create a pool for each cpu to send the batch_dist function to each split.
    # Then, close the pool and wait for jobs to complete before continuing:
    pool = mp.Pool(processes=n_jobs)
    queue_sum = sum(pool.map(batch_pdist, pairs_split, chunksize=N // n_jobs))
    pool.close()
    pool.join()
    N = weights.sum()
    # Compute the number of combinations, add to the number of unique pairs
    # and use that as the denominator to calculate the mean pairwise distance:
    mean = queue_sum / (((N - 1)**2 + (N + 1)) / 2 + N)
    # If you do not want to include distance from an item to itself use:
    # mean = queue_sum / (((N - 1)**2 + (N + 1)) / 2.0)

    return queue_sum, mean


if __name__ == "__main__":
    # Generate some data:
    N = 10
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

    # Pick some random floats for the counts/weights:
    counts = np.random.random_sample((N,)) * 10
    ##########################################################################
    # Parallel:
    # Parallelised code partially based on:
    # https://gist.github.com/baojie/6047780
    t = time()
    parallel_sum, parallel_mean = mean_pairwise_distance(X,
                                                 weights=counts,
                                                 n_jobs=min(mp.cpu_count(),N)
                                                         )
    print 'parallel:\t{} s'.format(time() - t)
    ##########################################################################

    ##########################################################################
    # Serial:
    # Comment this out if you use a high N as it will eat RAM!
    t = time()
    def dfun(u, v):
        return great_circle(u, v).km
    Y = pdist(X, metric=dfun)
    weights = [counts[i] * counts[j]
               for i in xrange(N - 1) for j in xrange(i + 1, N)]
    serial_sum = np.sum(weights * Y)
    N = counts.sum()
    serial_mean = serial_sum / (((N - 1)**2 + (N + 1)) / 2 + N)
    print 'serial:\t\t{} s'.format(time() - t)
    ##########################################################################

    # There is minor rounding error, but check for equality:
    assert np.round(serial_sum) == np.round(parallel_sum)
    assert np.round(serial_mean) == np.round(parallel_mean)
    print 'sum = {}'.format(parallel_sum)
    print 'mean = {}'.format(parallel_mean)
