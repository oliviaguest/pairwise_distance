from __future__ import division

import sklearn.datasets
import scipy.spatial.distance

import numpy as np
import multiprocessing as mp

from time import time
from itertools import combinations

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
def dispatch_jobs(X, w, job_number):
    N = X.shape[0]
    # Get the pairs and their weights to calculate the distances without needing
    # the whole of X:
    pairs = [(X[i:], X[:N - i], w[i:] * w[:N - i]) for i in xrange(1, N)]
    # Create slices of the pairs to send to each worker:
    pairs_slices = np.array_split(pairs, job_number)
    # weights_slices = np.array_split(pairs, job_number)

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
    return queue_sum

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

    # Parallel: ################################################################
    t = time()
    my_sum = dispatch_jobs(X, counts, mp.cpu_count())
    print 'parallel:\t{} s'.format(time() - t)
    ############################################################################

    # Serial: ##################################################################
    # Comment this out if you use a high N as it will eat RAM!
    # SERIOUSLY, be careful.
    t = time()
    Y = scipy.spatial.distance.pdist(X, 'euclidean')
    weights = [counts[i] * counts[j]
               for i in xrange(N - 1) for j in xrange(i + 1, N)]
    weighted_sum = np.sum(weights * Y)
    print 'serial:\t\t{} s'.format(time() - t)
    assert np.round(weighted_sum) == np.round(
        my_sum)  # There is minor rounding error after 8 decimal places.
    ############################################################################
    
    print 'sum = {}'.format(my_sum)
