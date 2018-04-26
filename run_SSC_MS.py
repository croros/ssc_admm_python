import numpy as np
import os
import time
import scipy.io as sio

datasetPath = ''

alpha = 800

maxNumGroup = 5

num = np.zeros((maxNumGroup, 1))

dataset = os.listdir(datasetPath)
numSamples = len(dataset)
missrateTotal1 = np.zeros((maxNumGroup,numSamples))
#Iterate through all samples in Hopkins155
for i in range(0,numSamples):
	if (dataset[i] not '.') and (dataset[i] not '..') and (os.path.isdir(dataset[i])):
		foundValidData = False #Just in case don't find anything
		sample = os.listdir(datasetPath+dataset[i]+'/')
        #For a given data sample, load info for that sample
		for j in range(0,len(sample):
			if '_truth.mat' in sample[i]:
				ind = j
				foundValidData = True
				break
				
		
		
		if foundValidData:
			#Now that we have the data, convert and pass into SSC
			sampleContents = sio.loadmat(datasetPath+dataset[i]+'/'+sample[ind])
			x = sampleContents('x') #Normalized homogen. coordinates of motion keypoints
			s = sampleContents('s') #ground truth labels
			n = np.max(s) #number of classes/groups
			N = np.size(x,1) #number of points
			F = np.size(x,2) #number of frames
			D = 2*F #Double for each element of point (x and y)
			X = np.reshape(np.moveaxis(x,[0,1,2],[0 2 1]).copy(),(D,N) ,order='F') #Reshape coordinates such that every two rows gives x and y coordinates for one frame (x in one row, y in other)
			
			#Set parameters for SSC call
			r = 0
			isAffine = True #If dealing with Affine subspaces
			hasOutlier = False #If data has outliers
			rho = 0.7 #relaxation variable used for determining how many rows in C to keep (after applying ADMM)
			
			missrate1,C1,grps1 = SSC(X,r,s,alpha,rho,isAffine,hasOutlier)
			
			#Keep track of missrate by class/group
			missrateTotal1[n-1,num[n-1]] = missrate1
			num[n-1] += 1
			
			
			np.savez(datasetPath+dataset[i]+'/SSC_MS',missrate1=missrate1,C1=C1,grps1=grps1)
			
avgmissrate1 = []
medmissrate1 = []		
for i in [1,2]:
	avgmissrate1[i-1] = np.mean(missrateTotal1[i,:])
	medmissrate1[i-1] = np.median(missrateTotal1[i,:])
	
np.savez('./ssc_mat_'+time.strftime('%Y_%m_%d_%H_%M_%S'),missrateTotal1=missratetotal1,avgmissrate1=avgmissrate1,medmissrate1=medmissrate1,alpha=alpha)