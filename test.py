# import dask.dataframe as dd
import dask.array as da
import numpy as np
import sklearn.datasets
# Generate some data:
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
        X = da.concatenate([X, np.concatenate((data, extra), axis=0)], axis=0)
    except:
        X = da.from_array(np.concatenate((data, extra), axis=0), chunks= (1000, 1000))
print X.shape
N = X.shape[0]
del data, extra, labels_true
# print dask_X
# pairs = [(X[i:], X[:N - i]) for i in xrange(1, N)]
# print len(pairs)
sums = []
counts = []
for i in xrange(1,N):                    # One million times
    chunk = X[i:]
    #  X[:N - i]    # Pull out chunk
    print i, chunk[0]
    # positive = chunk[chunk > 0]             # Filter out negative elements
    # sums.append(positive.sum())             # Sum chunk
    counts.append(chunk.size)            # Count chunk

result = delayed(sum)(counts)    # Aggregate results

result.compute()                            # Perform the computation
# print pairs.compute()
