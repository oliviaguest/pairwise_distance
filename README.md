# pairwise_distance

This code takes a set of 2D data points ```X``` and calculates the sum and the mean of the pairwise Euclidean distances between the points in parallel. 

In theory it is equivalent to the following (where ```N = X.shape[0]``` and ```counts``` is an array of length ```N``` with counts per ```X``` value):
``` python
    Y = scipy.spatial.distance.pdist(X, 'euclidean')
    weights = [counts[i] * counts[j]
               for i in xrange(N - 1) for j in xrange(i + 1, N)]
    serial_sum = np.sum(weights * Y)
    serial_mean = serial_sum / (((N - 1)**2 + (N + 1)) / 2 + N)
```

Importantly, however, it will not run out of memory for huge ```X```s (assuming X itself can fit into RAM).
Space complexity is constant. 

