from __future__ import division

import multiprocessing
import sklearn.datasets
import multiprocessing
import scipy.spatial.distance

import numpy as np

from time import time
from itertools import combinations


def do_job(job_id, data_slice, q):
    partial_sum = 0
    for item in data_slice:
        partial_sum += np.sum(np.sqrt(np.sum((item[0] - item[1])**2, axis=1)))
    q.put(partial_sum)


def dispatch_jobs(data, job_number):
    total = len(data)
    slices = np.array_split(data, job_number)
    jobs = []
    queues = []
    for i, s in enumerate(slices):
        queues.append(multiprocessing.Queue())

        j = multiprocessing.Process(target=do_job, args=(i, s, queues[i]))
        jobs.append(j)
    for j in jobs:
        j.start()
    queue_sum = 0
    for q in queues:
        queue_sum += q.get()
    print queue_sum
    return queue_sum


if __name__ == "__main__":
    ####### Generate some data ###############################################
    N = 100
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
    # Get the pairs to calculate the distances ###############################
    pairs = [(X[i:], X[:N - i]) for i in xrange(1, N)]
    # Do the thing ###########################################################
    # I have 32 cores, this gives good performance.
    my_sum = dispatch_jobs(pairs, multiprocessing.cpu_count() - 4)
    # Comment this out if you use a high N as it will eat RAM! ###############
    Y = scipy.spatial.distance.pdist(X, 'euclidean')
    print np.sum(Y)
    assert np.round(np.sum(Y)) == np.round(
        my_sum)  # there is minor rounding error

    # Backstory of ideas: ####################################################
    ##########################################################################
    # Brad's idea:
    # t = time()
    # pool = mp.Pool(4,maxtasksperchild=4)
    # results = []
    # pairs = [(X[i:], X[:N - i]) for i in xrange(0, N)]
    # results = pool.map(calculate_pairwise_distance_sum, pairs, chunksize=100)
    # pool.close()
    # print sum(results)
    # print '{} s'.format(time() - t)

    ##########################################################################
    # My idea:
    # t = time()
    # p = mp.Pool(12,maxtasksperchild=2,)
    # results = []
    # combs = [(X[comb[0]], X[comb[1]])
    #          for comb in combinations(xrange(X.shape[0]), 2)]
    # results = p.map(calculate_pairwise_distance_tuple, combs, chunksize=100)
    # print sum(results)
    # print '{} s'.format(time() - t)
