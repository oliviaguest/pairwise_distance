from __future__ import division

import sklearn.datasets
import scipy.spatial.distance

import numpy as np
import multiprocessing as mp

from time import time
from math import factorial
from itertools import combinations

##############################
# Code: Olivia Guest         #
# Algorithm: Bradley C. Love #
##############################

def do_job(data_slice, job_index, queue):
    # print job_index, data_slice.shape
    partial_sum = 0
    for i, points in enumerate(data_slice):
        # Each data_slice has tuples consisting of two points that we need to find the
        # Euclidean distance between and their weight:
        # points[2] are the weights for the pair points[0] and points[1]
        partial_sum += np.sum(points[2] *
                              np.sqrt(np.sum((points[0] - points[1])**2
                              , axis=1)))
    queue.put(partial_sum)

# If you want to memory profile this funtion to see it is roughly constant, feel
# free to comment in the decorator and run with memory_profiler (install it)
# as below:
# python -m memory_profiler pairwise_distance.py
# @profile
def mean_pairwise_distance(X, weights = None, n_jobs = None):
    """Function that returns the sum and mean of the pairwise distances of an 2D
    array X.

    Required arguments:
    X       --  2D array of points.

    Optional arguments:
    weights -- 1D array of counts or weights per point in X (default: 1s).
    n_jobs  -- Numper of cores to use for calulation (default: all).
    """
    N = X.shape[0]
    if weights is None:
        weights = np.ones((N,))
    if n_jobs is None:
        n_jobs = mp.cpu_count()
    # Get the pairs and their weights to calculate the distances without needing
    # the whole of X:
    pairs = [(X[i:], X[:N - i], weights[i:] * weights[:N - i])
             for i in xrange(1, N)]
    # Create slices of the pairs to send to each worker:
    pairs_slices = np.array_split(pairs, n_jobs)
    jobs = []
    queues = []
    for i, pairs_slice in enumerate(pairs_slices):
        # We need queues to get the data back from each worker function:
        queues.append(mp.Queue())
        # Create a job and append it to the list of jobs:
        jobs.append(mp.Process(
            target=do_job, args=(pairs_slice, i, queues[i])))
    # Start all the jobs:
    for j in jobs:
        j.start()
    # Calculate the sum:
    queue_sum = 0
    for q in queues:
        queue_sum += q.get()
    # Compute the number of combinations, add them to the number of unique pairs
    # and use that as the denominator to calculate the mean pairwise distance:
    mean = queue_sum / (((N - 1)**2 + (N + 1)) / 2 + N)
    # If you do not want to include distance from an item to itself use:
    # mean = queue_sum / (((N - 1)**2 + (N + 1)) / 2.0)
    return queue_sum, mean

if __name__ == "__main__":
    # Generate some data:
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

    # Pick some random floats for the counts/weights:
    counts = np.random.random_sample((N,)) * 10
    ############################################################################
    # Parallel:
    # Parallelised code partially based on:
    # https://gist.github.com/baojie/6047780
    t = time()
    parallel_sum, parallel_mean = mean_pairwise_distance(X,
                                                         weights = counts,
                                                         n_jobs = mp.cpu_count()
                                                        )
    print 'parallel:\t{} s'.format(time() - t)
    ############################################################################

    ############################################################################
    # Serial:
    # Comment this out if you use a high N as it will eat RAM!
    t = time()
    Y = scipy.spatial.distance.pdist(X, 'euclidean')
    weights = [counts[i] * counts[j]
               for i in xrange(N - 1) for j in xrange(i + 1, N)]
    serial_sum = np.sum(weights * Y)
    serial_mean = serial_sum / (((N - 1)**2 + (N + 1)) / 2 + N)
    print 'serial:\t\t{} s'.format(time() - t)
    ############################################################################

    # There is minor rounding error, but check for equality:
    assert np.round(serial_sum) == np.round(parallel_sum)
    assert np.round(serial_mean) == np.round(parallel_mean)
    print 'sum = {}'.format(parallel_sum)
    print 'mean = {}'.format(parallel_mean)
