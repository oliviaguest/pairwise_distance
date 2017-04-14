from __future__ import division
import multiprocessing as mp

import numpy as np

##############################
# Code: Olivia Guest         #
# Algorithm: Bradley C. Love #
##############################


def batch_pdist(data_slice):
    # Each data_slice has tuples consisting of two points that we need to
    # find the Euclidean distance between and their weight:
    # points[2] are the weights for the pair points[0] and points[1]
    return np.sum(np.dot(weights, np.sqrt(np.sum((X - Y)**2, axis=1)))
                  for X, Y, weights in data_slice)

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
        n_jobs = mp.cpu_count()

    # Get the pairs and their weights to calculate the distances without
    # needing the whole of X, split it into roughly equal sub-arrays per cpu:
    pairs_split = np.array_split([(X[i:], X[:N-i], weights[i:] * weights[:N-i])
                                 for i in xrange(1, N)],
                                 n_jobs, axis=axis)

    # Create a pool for each cpu to send the batch_dist function to each split.
    # Then, close the pool and wait for jobs to complete before continuing:
    pool = mp.Pool(processes=n_jobs)
    queue_sum = sum(pool.map(batch_pdist, pairs_split, chunksize=N//n_jobs))
    pool.close()
    pool.join()

    # Compute the number of combinations, add to the number of unique pairs
    # and use that as the denominator to calculate the mean pairwise distance:
    N = weights.sum()
    mean = queue_sum / (((N - 1)**2 + (N + 1)) / 2 + N)
    # If you do not want to include distance from an item to itself use:
    # mean = queue_sum / (((N - 1)**2 + (N + 1)) / 2.0)

    return queue_sum, mean
