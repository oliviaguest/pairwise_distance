##############################################################
# Code written by Frank Kanayet Camacho (@franink on github) #
##############################################################

# Parallel distance functions
# dataTask is the block of coordinates to work on and conditions for loop (start,end,step_size) where step_size 1 processes every item and step size size 2 every other, etc.

def ByClusterDistParallelHelper(dataTask):
	dataSegment=dataTask[0]
	helperDist=0
	# print("dataTask",dataTask[1:4])
	peopleInCluster=len(dataSegment)
	for j in xrange(dataTask[1],dataTask[2],dataTask[3]):
		helperDist+=sum(np.sqrt(np.add.reduce((dataSegment.iloc[(j+1):peopleInCluster]-dataSegment.iloc[j])**2,1)))
	return helperDist

#############################
# This function should be ready for prime time use in project
### what is the structure of data??? data should be called with M
def ByClusterDistParallel(data,step_size,label_name,numClusters,RealpeoplePerCluster):
    #some defintions
    pool = multiprocessing.Pool(None) #this defines the pool of processors/cores.
    #  the parameter says how many processes to use. "None" defaults to number defined by the system. on my mac, it also counts virtual cores (hyperthreading).
    # print "number of processes: ",pool._processes
    ######### Need to change data.num for appropriate varaible name for number of clusters in principle is numClusters
    peoplePerCluster=np.zeros(numClusters,int) #number of people in each cluster
    totalDist=0
    ############### Same here numClusters

    totalDistPerCluster=np.zeros(numClusters,float) #sum of pairwise distances for each cluster
    ################ same in next line... numClusters
    clusters = data[label_name].unique()
    for i, clus_index in enumerate(clusters): #This line used to say the below, but that is a bug, as congressional districts can have labels [1, 2] not only [0, 1]
    #for i in xrange(numClusters): # loop by cluster
        # print "Starting Cluster #",i
        ################ data.people needs to be changed by the list of xy coords
        ################ I actually have a way to do this with pandas
        temp = data.loc[data[label_name]==clus_index,'x':'y']

        temp=temp.iloc[range(0,len(temp),step_size)] #**** try shrinking the data matrix here by just discarding some data.
        peoplePerCluster[i]=len(temp) ## effective peoperPerCLuster after trimming
        #############    	create split of tasks for multiprocessing.
        tasks=[]
        taskSize=peoplePerCluster[i]/pool._processes
        for j in xrange(pool._processes):
            tasks.append([temp,j*taskSize,(j+1)*taskSize,1]) #step_size removed.
        tasks[j][2]=peoplePerCluster[i]-1 	#this ensures the last task gets any extra people resulting from dividing by pool.processes
        ############
        results = []
        r = pool.map_async(ByClusterDistParallelHelper, tasks, callback=results.append) #splits up the tasks across processors.
        r.wait() # Wait on the results
        totalDist+=np.sum(results) #this will be removed.
        totalDistPerCluster[i]=np.sum(results)
    # print totalDistPerCluster, peoplePerCluster
    avgDistPerCluster=(totalDistPerCluster)/(peoplePerCluster*(peoplePerCluster-1)/2.)  #step_size no longer in numerator
    # print "NOTE: STEP_SIZE REDUCES EFFECTIVE COUNTS:",step_size
    # print "Average Distance for each cluster:",avgDistPerCluster
    # print "People per cluster:",peoplePerCluster, "Real people per cluster:", np.asarray(RealpeoplePerCluster)
    # print "Overall Average Distance:",np.inner(avgDistPerCluster,peoplePerCluster)/np.sum(peoplePerCluster)
    return avgDistPerCluster, peoplePerCluster
