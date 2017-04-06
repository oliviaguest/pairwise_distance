# pairwise_distance

This code takes a set of data points ```X``` and calculates the sum of the pairwise Euclidean distances. 
In theory it is equivalent to the following:
``` python
np.sum(scipy.spatial.distance.pdist(X, 'euclidean'))
```
Importantly, however, it will not run out of memory for huge ```X```s.
It's time complexity is higher than Scipy's version but it's space complexity is constant. 

It probably could do with some more work to tidy it up, etc. — apologies. 
