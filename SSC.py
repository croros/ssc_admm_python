import numpy as np
import sys

def SSC(X,r,isAffine,alpha,hasOutlier,rho,s):

	numArgs = len(sys.argv)

	if numArgs < 6:
		rho = 1

	if numArgs < 5:
		hasOutlier = False

	if numArgs < 4:
		alpha = 20

	if numArgs < 3:
		isAffine = True

	if numArgs < 2:
		r = 0
		
		
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
	