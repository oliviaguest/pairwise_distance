from __future__ import division
from time import time
from math import factorial
from itertools import combinations
import multiprocessing as mp

import sklearn.datasets
import scipy.spatial.distance
import numpy as np

##############################
# Code: Olivia Guest         #
# Algorithm: Bradley C. Love #
##############################


def do_job(data_slice, queue):
    # Each data_slice has tuples consisting of two points that we need to
    # find the Euclidean distance between and their weight:
    # points[2] are the weights for the pair points[0] and points[1]
    slice_sum = np.sum(np.dot(weights, np.sqrt(np.sum((X - Y)**2, axis=1)))
                       for X, Y, weights in data_slice)
    queue.put(slice_sum)

# If you want to memory profile this funtion to see it is roughly constant,
# feel free to comment in the decorator and run with memory_profiler (install
# it) as below:
# python -m memory_profiler pairwise_distance.py
# @profile


def mean_pairwise_distance(X, weights=None, n_jobs=None, axis=0):
    """Function that returns the sum and mean of the pairwise distances of an 2D
    array X.

    Required arguments:
    X       --  2D array of points.

    Optional arguments:
    weights -- 1D array of counts or weights per point in X (default: 1s).
    n_jobs  -- Number of cores to use for calulation (default: all).
    axis    -- The axis of X corresponding to data elements (default: 0).
    """
    N = X.shape[axis]
    if weights is None:
        weights = np.ones((N,))
    if n_jobs is None:
        n_jobs = mp.cpu_count()

    # Get the pairs and their weights to calculate the distances without
    # needing the whole of X:
    pairs = [(X[i:], X[:N - i], weights[i:] * weights[:N - i])
             for i in xrange(1, N)]
    # Create approximately equal splits of the pairs for each cpu:
    pairs_slices = np.array_split(pairs, n_jobs)

    # Make queues to collect results and processes to do the calculations:
    queues = [mp.Queue() for i in xrange(n_jobs)]
    jobs = [mp.Process(target=do_job, args=(pairs_slice, queue))
            for queue, pairs_slice in zip(queues, pairs_slices)]

    # Start all the jobs, aggregate the sum when they finish:
    for j in jobs:
        j.start()
    queue_sum = sum(q.get() for q in queues)

    # Compute the number of combinations, add to the number of unique pairs
    # and use that as the denominator to calculate the mean pairwise distance:
    mean = queue_sum / (((N - 1)**2 + (N + 1)) / 2 + N)
    # If you do not want to include distance from an item to itself use:
    # mean = queue_sum / (((N - 1)**2 + (N + 1)) / 2.0)
    return queue_sum, mean

if __name__ == "__main__":
    # Generate some data:
    N = 1000
    np.random.seed(10)
    n_jobs = mp.cpu_count()
    n_samples = N * 3 // 4  # same as floor(3/4 * N)

    # Blob set 1
    centers = [[0., 0.],
               [1., 0.],
               [0.5, np.sqrt(0.75)]]
    n_clusters = len(centers)
    cluster_std = [0.3] * n_clusters
    data, labels_true = sklearn.datasets.make_blobs(n_samples=n_samples,
                                                    centers=centers,
                                                    cluster_std=cluster_std)

    # Blob set 2
    centers = [[0.5, np.sqrt(0.75)]]
    n_clusters = len(centers)
    cluster_std = [0.3] * n_clusters
    extra, labels_true = sklearn.datasets.make_blobs(n_samples=N - n_samples,
                                                     centers=centers,
                                                     cluster_std=cluster_std)
    X = np.concatenate((data, extra), axis=0)

    # Pick some random floats for the counts/weights:
    counts = np.random.random_sample((N,)) * 10
    ##########################################################################
    # Parallel:
    # Parallelised code partially based on:
    # https://gist.github.com/baojie/6047780
    t_start = time()
    parallel_sum, parallel_mean = mean_pairwise_distance(X,
                                                         weights=counts,
                                                         n_jobs=n_jobs)
    t_end = time()
    print 'parallel:\t{} s'.format(t_end - t_start)
    ##########################################################################

    ##########################################################################
    # Serial:
    # Comment this out if you use a high N as it will eat RAM!
    t_start = time()
    weights = np.array([counts[i] * counts[j]
                       for i in xrange(N - 1) for j in xrange(i + 1, N)])
    serial_sum = np.dot(weights, scipy.spatial.distance.pdist(X, 'euclidean'))
    serial_mean = serial_sum / (((N - 1)**2 + (N + 1)) / 2 + N)
    t_end = time()
    print 'serial:\t\t{} s'.format(t_end - t_start)
    ##########################################################################

    # There is minor rounding error, but check for equality:
    assert np.round(serial_sum) == np.round(parallel_sum)
    assert np.round(serial_mean) == np.round(parallel_mean)
    print 'sum = {}'.format(parallel_sum)
    print 'mean = {}'.format(parallel_mean)
