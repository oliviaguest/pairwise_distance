import multiprocessing
import subprocess
from math import exp,sqrt


# this is just a function that wastes some time doing math computations. 
def calculate(value):
	print("in: ",value)
	for i in range(1000000):
		sqrt(exp(1./(value[0]+.78)))
	return value[0]+value[1]


#you can use this to either split on the parameter (bad) or to split up runs of stoachastic model across processors (good/easy). or these could be different chanins running in parallel, etc.
def central():
    pool = multiprocessing.Pool(None) #this defines the pool of processors/cores. the parameter says how many processes to use. "None" defaults to number defined by the system. on my mac, it also counts virtual cores (hyperthreading).
    #tasks = range(100)	#tasks to split up across processors.
    tasks=[[1,3],[2,4]]
    results = [] 
    r = pool.map_async(calculate, tasks, callback=results.append) #splits up the tasks across processors.
    r.wait() # Wait on the results
    return results
    
def main():
	print("results: ",central())
	
	
###########################################################
# let's start
###########################################################

if __name__ == '__main__':
    main()
	
	