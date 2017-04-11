# pairwise_distance

## Usage
This code takes a set of 2D data points ```X``` and calculates the sum and the mean of the pairwise Euclidean distances between the points in parallel. 
To call use (```weights``` and ```n_jobs``` are optional): 
``` python
parallel_sum, parallel_mean = mean_pairwise_distance(X,
                                                     weights = how_to_weight_each_X,
                                                     n_jobs = how_many_cores_to_use)
```

In theory it is equivalent to the following (where ```N = X.shape[0]``` and ```counts``` is an array of length ```N``` with counts per ```X``` value):
``` python
    Y = scipy.spatial.distance.pdist(X, 'euclidean')
    weights = [counts[i] * counts[j]
               for i in xrange(N - 1) for j in xrange(i + 1, N)]
    serial_sum = np.sum(weights * Y)
    serial_mean = serial_sum / (((N - 1)**2 + (N + 1)) / 2 + N)
```

Importantly, however, it will not run out of memory for huge ```X```s (assuming ```X``` itself can fit into RAM).
Space complexity is constant.

## Tests
Testing this module using `pytest` and `memory_profiler` can be easily invoked using `make`.
* `make test` will run the basic assertion tests.
* `make memory` will run the memory profiler and display a plot of the processes.
* `make save_plot` will save a plot of the most recent memory profiling data as a png.
* `make clean` will delete all the data files created by the memory profiler.

## Authors
* __Code__: Olivia Guest [@oliviaguest](http://github.com/oliviaguest) [oliviaguest.com](http://oliviaguest.com)
* __Algorithm__: Bradley C. Love [@lovebc](http://github.com/lovebc) [bradlove.org](http://bradlove.org)
