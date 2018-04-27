import numpy as np
import sys

def SSC(X,r=0,isAffine=False,alpha=20,hasOutlier=False,rho=1,s):

	n = np.max(s) #Number of groups moving (ground truth)
	#Xp = DataProjection(X,r) #Use PCA to project onto subspace r (not used for motion segmentation
	Xp = X
	
	#Perform ADMM to get the sparse coefficient matrix C (for sparse linear combinations)
	if not hasOutlier:
		Cout = admmLasso(Xp,isAffine,alpha)
		Ckeep = Cout #Use this C for building graph (different if have outliers)
	else:
		Cout = admmOutlier(Xp,affine,alpha) #TODO: Implement
		N = np.size(Xp,1)
		Ckeep = Cout[0:N-1,:] #Only keep top N rows if there are outliers
	
	#Get similarity matrix by forming a symmetric adjacency matrix
	W = getSymmAdj(threshC(Ckeep,rho))
	#Apply spectral clustering to graph created from W to group data points
	grps = spectralClustering(W,n)
	#Compare with ground truth to determine missclassification
	missrate = Misclasification(grps,s)
	
	return missrate, Cout, grps

def admmLasso(Y,isAffine=False,alpha=800,thresh=0.0002,maxIter=200):
	
	#Check if alpha is a single value or multiple
	if isinstance(alpha,int) or isinstance(alpha,float):
		alpha1 = alpha #Used for lambda_z (penalty/regularization parameter of noise)
		alpha2 = alpha #Used for rho (penalty parameter/step size used for updating the lagrange multipliers deltaBig and deltaLil )
	else:
		alpha1 = alpha[0]
		alpha2 = alpha[1]
		
	if isinstance(thresh,int) or isinstance(thresh,float):
		thresh1 = thresh #Threshold on maximum error between Auxiliary variable A and original variable C (goal is A=C)
		
		#used for initial L2-norm error of Y-Y*A (i.e. E^k-E^{k-1}). 
		#E is the outlier coefficient matrix, so its not really needed here and its recorded for sake of interest.
		thresh2 = thresh 
	
	N = np.size(Y,1)
	
	#setting penalty parameters for the ADMM
	lambda_z = alpha1/(getMu(Y))
	rho = alpha2
	
	#Further computations need us to distinguish linear and affine subspaces
	if isAffine:
		firstA = np.linalg.inv(lambda_z*(dot(Y.T,Y))+rho*(np.eye(N))+rho*np.ones((N,N))#First part of A (doesn't change so precompute)
		C_k = np.zeros((N,N))
		deltaBig = np.zeros((N,N))
		deltaLil = np.zeros((1,N))
		
		errIdx = 0
		errAC = np.zeros((1,maxIter))
		errAC[errIdx] = 10*thresh1
		errE = np.zeros((1,maxIter))
		errE[errIdx] = 10*thresh2
		errAt1 = np.zeros((1,maxIter))
		errAt1[errIdx] = 10*thresh1
		
		#Now that parameters are set up, begin ADMM iterations
		
		
		
		
		
		
		
		
		
	