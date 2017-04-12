from __future__ import division
from time import time
import multiprocessing as mp

from scipy.spatial.distance import pdist
from sklearn.datasets import make_blobs
import numpy as np

# This workaround (read: hack) makes pytest and memory_profiler play nice:
from os import path
import sys
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from pairwise_distance import mean_pairwise_distance


def generate_data(N, seed=10):
    """ This generates some test data that we can use to test our pairwise-
    distance functions.

    Required arguments:
    N       -- The number of datapoints in the test data.

    Optional arguments:
    seed    -- The seed for NumPy's random module.
    """

    # Generate some data:
    np.random.seed(seed)
    n_samples1 = N * 3 // 4  # same as floor(3/4 * N)
    n_samples2 = N - n_samples1

    # Blob set 1
    centers1 = [[0., 0.],
                [1., 0.],
                [0.5, np.sqrt(0.75)]]
    cluster_std1 = [0.3] * len(centers1)
    data, _ = make_blobs(n_samples=n_samples1,
                         centers=centers1,
                         cluster_std=cluster_std1)

    # Make sure Blob 1 checks out

    # Blob set 2
    centers2 = [[0.5, np.sqrt(0.75)]]
    cluster_std2 = [0.3] * len(centers2)
    extra, _ = make_blobs(n_samples=n_samples2,
                          centers=centers2,
                          cluster_std=cluster_std2)

    return np.concatenate((data, extra), axis=0)


def test_mean_pairwise_distance(N=1000):
    """
    This function computes the pairwise distances on a (small) simulated
    dataset to make sure the distributed function returns the same sum and
    mean for pairwise distances as SciPy's pdist function.

    Optional arguments:
    N -- The number of points to use in the simulated dataset.
    """

    # Generate data and some random floats for the weights:
    X = generate_data(N)
    weights = np.random.random_sample((N,)) * 10

    ##########################################################################
    # Parallel:
    # Parallelised code partially based on:
    # https://gist.github.com/baojie/6047780
    t_start_parallel = time()
    parallel_sum, parallel_mean = mean_pairwise_distance(X, weights=weights)
    t_end_parallel = time()
    ##########################################################################

    ##########################################################################
    # Serial:
    # Comment this out if you use a high N as it will eat RAM!
    t_start_serial = time()
    weights = np.array([weights[i] * weights[j]
                       for i in xrange(N - 1) for j in xrange(i + 1, N)])
    serial_sum = np.dot(weights, pdist(X, 'euclidean'))
    serial_mean = serial_sum / (((N - 1)**2 + (N + 1)) / 2 + N)
    t_end_serial = time()
    ##########################################################################

    # There is minor rounding error, but check for equality:
    assert np.isclose(serial_sum, parallel_sum)
    assert np.isclose(serial_mean, parallel_mean)

    # Print out a nice summary of the performance and data measures:
    def print_time(s, t):
        return "%10s: %10.6f s" % (s, t)
    print  # Print a newline to make things nice, due to pytest
    print print_time('parallel', t_end_parallel - t_start_parallel)
    print print_time('serial', t_end_serial - t_start_serial)
    print 'sum = {}'.format(parallel_sum)
    print 'mean = {}'.format(parallel_mean)

# This file shouldn't be executed or imported on its own. This __main__ block
# is so that external tools (e.g. memory_profiler) work correctly:
if __name__ == "__main__":
    test_mean_pairwise_distance()
