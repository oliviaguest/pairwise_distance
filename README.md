# pairwise_distance

This code takes a set of data points ```X``` and calculates the sum of the pairwise Euclidean distances. 
In theory it is equivalent to the following:
```
np.sum(scipy.spatial.distance.pdist(X, 'euclidean'))
```
Importantly, however, it will not run out of memory for huge ```X```s. 
