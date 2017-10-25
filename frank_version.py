##############################################################
# Code written by Frank Kanayet Camacho (@franink on github) #
##############################################################

# Parallel distance functions
# dataTask is the block of coordinates to work on and conditions for loop
# (start,end,step_size) where step_size 1 processes every item and step
# size size 2 every other, etc.


def ByClusterDistParallelHelper(dataTask):
    dataSegment = dataTask[0]
    helperDist = 0
    # print("dataTask",dataTask[1:4])
    pointsInCluster = len(dataSegment)
    for j in xrange(dataTask[1], dataTask[2], dataTask[3]):
        helperDist += sum(np.sqrt(np.add.reduce(
            (dataSegment.iloc[(j + 1):pointsInCluster] - dataSegment.iloc[j])**2, 1)))
    return helperDist

#############################
# This function should be ready for prime time use in project
# what is the structure of data??? data should be called with M


def ByClusterDistParallel(data, step_size, label_name, numClusters, RealpointsPerCluster):
    # some defintions
    # this defines the pool of processors/cores.
    pool = multiprocessing.Pool(None)
    # number of points in each cluster
    pointsPerCluster = np.zeros(numClusters, int)
    totalDist = 0
    # sum of pairwise distances for each cluster
    totalDistPerCluster = np.zeros(numClusters, float)
    # same in next line... numClusters
    clusters = data[label_name].unique()
    for i, clus_index in enumerate(clusters):
        temp = data.loc[data[label_name] == clus_index, 'x':'y']

        # **** try shrinking the data matrix here by just discarding some data.
        temp = temp.iloc[range(0, len(temp), step_size)]
        pointsPerCluster[i] = len(temp)
        # create split of tasks for multiprocessing.
        tasks = []
        taskSize = pointsPerCluster[i] / pool._processes
        for j in xrange(pool._processes):
            # step_size removed.
            tasks.append([temp, j * taskSize, (j + 1) * taskSize, 1])
        # this ensures the last task gets any extra points resulting from
        # dividing by pool.processes
        tasks[j][2] = pointsPerCluster[i] - 1
        ############
        results = []
        # splits up the tasks across processors.
        r = pool.map_async(ByClusterDistParallelHelper,
                           tasks, callback=results.append)
        r.wait()  # Wait on the results
        totalDist += np.sum(results)  # this will be removed.
        totalDistPerCluster[i] = np.sum(results)
    avgDistPerCluster = (totalDistPerCluster) / (pointsPerCluster *
                                                 (pointsPerCluster - 1) / 2.)  # step_size no longer in numerator
    return avgDistPerCluster, pointsPerCluster
