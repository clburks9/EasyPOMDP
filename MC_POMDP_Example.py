from __future__ import division
from sys import path

#path.append('../../src/');
from gaussianMixtures import GM, Gaussian 
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
import numpy as np
import copy
import matplotlib.pyplot as plt
#Exploring Monte Carlo POMDP methods

#Step 1: Build a working particle filter



#Algorithm in Probabilistic Robotics by Thrun, page 98
def particleFilter(Xprev,a,o,pz):
	Xcur = []; 

	delA = [-1,1,0]; 
	delAVar = 0.5;
	allW = []; 
	allxcur = []; 
	for xprev in Xprev:
		xcur = np.random.normal(xprev+delA[a],delAVar,size =1)[0].tolist(); 
		
		w = pz[o].pointEval(xcur); 
 
		allW.append(w);
		allxcur.append(xcur); 

	#normalize weights for kicks
	suma = 0; 
	for w in allW:
		suma+=w; 
	for i in range(0,len(allW)):
		allW[i] = allW[i]/suma; 

	for m in range(0,len(Xprev)):
		c = np.random.choice(allxcur,p=allW); 
		Xcur.append(copy.deepcopy(c)); 
	return Xcur; 


def beliefUpdate(b,a,o,pz):
		btmp = GM(); 

		adelA = [-1,1,0]; 
		adelAVar = 0.5;

		for obs in pz[o].Gs:
			for bel in b.Gs:
				sj = np.matrix(bel.mean).T; 
				si = np.matrix(obs.mean).T; 
				delA = np.matrix(adelA[a]).T; 
				sigi = np.matrix(obs.var); 
				sigj = np.matrix(bel.var); 
				delAVar = np.matrix(adelAVar); 

				weight = obs.weight*bel.weight; 
				weight = weight*mvn.pdf((sj+delA).T.tolist()[0],si.T.tolist()[0],np.add(sigi,sigj,delAVar)); 
				var = (sigi.I + (sigj+delAVar).I).I; 
				mean = var*(sigi.I*si + (sigj+delAVar).I*(sj+delA)); 
				weight = weight.tolist(); 
				mean = mean.T.tolist()[0]; 
				var = var.tolist();
				 

				btmp.addG(Gaussian(mean,var,weight)); 


		btmp.normalizeWeights(); 
		btmp.condense(1); 
		btmp.normalizeWeights(); 


		return btmp; 


#Defined (sloppily) as the average distance of every 1 particle from every 2 particle
def particleSetDistance(X1,X2):
	
	suma = 0; 
	for i in range(0,len(X1)):
		for j in range(0,len(X2)):
			suma += abs(X1[i] - X2[j]); 
	aveDist = suma/(len(X1)*len(X2)); 

	return aveDist; 


def shepardsInterpolation(V,XPRIME,al = .1,retDist=False):
	
	dists = [0]*len(V); 

	#Upper triangular
	for i in range(0,len(V)):
			dists[i] = particleSetDistance(V[i][0],XPRIME); 

	#find points within a threshold
	distSort = sorted(set(dists));

	if(retDist == False):
		thres = 1; 
		cut = 1; 
		for i in range(0,len(distSort)):
			if(distSort[i]>1):
				cut = i; 
				break; 
		distSort = distSort[:cut];  

	eta = 0; 
	suma = 0; 
	for i in range(0,len(distSort)):
		eta += 1/dists[dists.index(distSort[i])]; 
		suma += (1/dists[dists.index(distSort[i])])*V[dists.index(distSort[i])][1]; 

	if(eta>0):	
		eta = 1/eta; 
	ans = eta*suma; 

	if(retDist):
		return [distSort,dists,eta]
	else:
		return ans; 


#Algorithm in Probabilistic Robotics by Thrun, page 560
def MCPOMDP(b0,M=100,iterations=100,episodeLength=10):
	V=[]; 
	delA = [-1,1,0]; 
	delAVar = 0.1;
	R = GM(13,0.1,1); 
	simLoopsN = 10; 
	gamma = .9; 

	#learning param
	alpha = 0.1; 

	pz = [GM(),GM(),GM()]; 
	for i in range(0,12):
		pz[0].addG(Gaussian(i,1,1)); 
	pz[1].addG(Gaussian(13,1,1)); 
	for i in range(14,21):
		pz[2].addG(Gaussian(i,1,1));


	#until convergence or time
	for count in range(0,iterations):
		print(count); 
		#sample x from b
		#[mean,var] = (b0.getMeans()[0],b0.getVars()[0])
		#x = np.random.normal(mean,var); 

		#sample particle set from b 
		X = b0.sample(M); 
		
		#for each episode?
		for l in range(0,episodeLength):
			part = np.random.choice(X); 
			Q = [0]*len(delA); 
			#for each action
			for a in range(0,len(delA)):
				#Simulate possible new beliefs
				for n in range(0,simLoopsN):
					x = part; 
					xprime = np.random.normal(x+delA[a],delAVar,size =1)[0].tolist(); 
					ztrial = [0]*len(pz); 
					for i in range(0,len(pz)):
						ztrial[i] = pz[i].pointEval(xprime); 
					z = ztrial.index(max(ztrial)); 
					XPRIME = particleFilter(X,a,z,pz); 
					Q[a] = Q[a] + (1/simLoopsN)*gamma*(R.pointEval(xprime) + shepardsInterpolation(V,XPRIME)); 

			[distSort,dists,eta] = shepardsInterpolation(V,X,retDist=True); 
			#update used value entries
			for i in range(0,len(distSort)):
				tmpVal = V[dists.index(distSort[i])][1] + alpha*eta*(1/dists[dists.index(distSort[i])])*(max(Q)-V[dists.index(distSort[i])][1]);
				#V[dists.index(distSort[i])] = [V[dists.index(distSort[i])][0],tmpVal,Q.index(max(Q))]; 
				V[dists.index(distSort[i])] = [V[dists.index(distSort[i])][0],tmpVal,V[dists.index(distSort[i])][2]]; 

			act = Q.index(max(Q)); 
			V.append([X,max(Q),act]); 



			xprime = np.random.normal(x+delA[act],delAVar,size =1)[0].tolist(); 
			ztrial = [0]*len(pz); 
			for i in range(0,len(pz)):
				ztrial[i] = pz[i].pointEval(xprime); 
			z = ztrial.index(max(ztrial));
			Xprime = particleFilter(X,act,z,pz); 
			x = xprime; 
			X =	copy.deepcopy(Xprime); 

	return V




def testParticleFilter():
	pz = [GM(),GM(),GM()]; 
	for i in range(-5,0):
		pz[0].addG(Gaussian(i,1,1)); 
	pz[1].addG(Gaussian(0,1,1)); 
	for i in range(0,5):
		pz[2].addG(Gaussian(i,1,1)); 
	initAct= GM(0,0.5,1); 
	actSeq = [0,0,0,0,1,1,1,1,1,1,2]; 
	obsSeq = [1,0,0,0,0,0,1,1,2,2,2]; 
	numParticles = 100; 

	initPart = []; 
	for i in range(0,numParticles):
		initPart.append(np.random.normal(0,0.5)); 

	seqPart = []; 
	seqPart.append([initPart]); 
	seqAct = []; 
	seqAct.append(initAct); 
	for i in range(0,len(actSeq)):
		
		tmp = particleFilter(seqPart[i][0],actSeq[i],obsSeq[i],pz); 
		seqPart.append([tmp]); 
		
		tmp = beliefUpdate(seqAct[i],actSeq[i],obsSeq[i],pz)

		seqAct.append(tmp); 

	allSigmasAct = []; 
	allSigmasPart = []; 
	for i in range(0,len(seqPart)):
		mean = 0; 
		var = 0; 
		for j in range(0,len(seqPart[i])):
			mean+=seqPart[i][0][j]/len(seqPart[i]); 
		for j in range(0,len(seqPart[i][0])):
			var += (seqPart[i][0][j]-mean)*(seqPart[i][0][j]-mean)/len(seqPart[i][0]); 
		allSigmasPart.append(np.sqrt(var)); 
		allSigmasAct.append(np.sqrt(seqAct[i].getVars()[0])); 

	for i in range(0,len(allSigmasAct)):
		while isinstance(allSigmasAct[i],list) or isinstance(allSigmasAct[i],np.ndarray):
			allSigmasAct[i] =allSigmasAct[i].tolist()[0][0]
	diffs = []; 
	ratios = [];
	averageDiff=0; 
	averageRatio = 0; 
	for i in range(0,len(allSigmasAct)):
		diffs.append(allSigmasPart[i]-allSigmasAct[i]); 
		ratios.append(allSigmasPart[i]/allSigmasAct[i]); 
		averageDiff += diffs[i]/len(allSigmasAct); 
		averageRatio += ratios[i]/len(allSigmasAct); 

	fig,axarr = plt.subplots(len(actSeq),1); 
	for i in range(0,len(actSeq)):
		[x,c] = seqAct[i].plot(low=-5,high=5,vis=False); 
		axarr[i].plot(x,c,color='r'); 
		axarr[i].hist(seqPart[i],normed=1,bins=10); 
		axarr[i].set_xlim([-5,5]);
		#print(str(i) + ' Part Sig:' + str(allSigmasPart[i]) + '  Act Sig:' + str(allSigmasAct[i]) + ' Diff:' + str(allSigmasPart[i]-allSigmasAct[i]));

	print('Average Sigma Difference: ' + str(averageDiff)); 
	print('Average Sigma Ratio: ' + str(averageRatio)); 
	plt.show(); 


def displayPolicy(V):
	plt.scatter(0,0.15,c='g'); 
	plt.scatter(0,0.15,c='r'); 
	plt.scatter(0,0.15,c='b');

	plt.legend(['Right','Stay','Left']); 


	for v in V:
		if(v[2]==0):
			col = 'b'; 
		elif(v[2]==1):
			col='g'; 
		else:
			col='r'; 
		plt.hist(v[0],normed=1,bins=10,color=col);
	
	plt.title('MC-POMDP Policy with 100 Particles, Goal = 13')
	plt.xlabel('Position')
	plt.show(); 

def testMCPOMDP():
	b = GM();
	for i in range(0,21):
		b.addG(Gaussian(i,0.5,1)); 

	V= MCPOMDP(b,M=15,iterations=20,episodeLength=4); 
	
	#The best one is 4
	f = open('MCPolicy1.npy',"w");
	np.save(f,V); 

	numParticles = 100; 
	testSet = [0]*21; 
	acts = [0]*21; 
	for i in range(0,len(testSet)):
		testSet[i] = GM(i,1,1).sample(numParticles); 
		[distSort1,dists1,eta1] = shepardsInterpolation(V,testSet[i],retDist=True); 
		acts[i] = V[dists1.index(distSort1[0])][2]; 

	for i in range(0,len(testSet)):
		print(i,acts[i]); 


	'''
	plt.hist(testX1,normed=1,bins=10);
	plt.hist(testX2,normed=1,bins=10);
	plt.hist(testX3,normed=1,bins=10);
	plt.show(); 

	displayPolicy(V); 
	'''
	
def compareToMDP(V):
	grid = np.ones(shape=(1,21)); 

	plt.figure();
	plt.scatter(-5,.5,c='k'); 
	plt.scatter(-5,.5,c='r');
	plt.scatter(-5,.5,c='y');

	
	numParticles = 1; 
	testSet = [0]*21; 
	acts = [0]*21; 
	for i in range(0,len(testSet)):
		testSet[i] = GM(i,0.001,1).sample(numParticles); 
		[distSort1,dists1,eta1] = shepardsInterpolation(V,testSet[i],retDist=True); 
		grid[0][i] = V[dists1.index(distSort1[0])][2]; 

	plt.xlim([0,20]); 
	plt.imshow(grid,extent=[0,20,0,1],cmap='inferno');
	plt.legend(['Left','Right','Stay']); 
	plt.title('Comparison to MDP policy implementation'); 

	plt.show(); 

def checkBoth(V):
	fig,axarr = plt.subplots(2,sharex=True); 

	axarr[0].scatter(0,0.15,c='g'); 
	axarr[0].scatter(0,0.15,c='r'); 
	axarr[0].scatter(0,0.15,c='b');

	axarr[0].legend(['Right','Stay','Left']); 


	for v in V:
		if(v[2]==0):
			col = 'b'; 
		elif(v[2]==1):
			col='g'; 
		else:
			col='r'; 
		axarr[0].hist(v[0],normed=1,bins=10,color=col);
	
	axarr[0].set_title('MC-POMDP Policy with 100 Particles')
	
	grid = np.ones(shape=(1,21)); 

	
	axarr[1].scatter(-5,.5,c='k'); 
	axarr[1].scatter(-5,.5,c='r');
	axarr[1].scatter(-5,.5,c='y');

	
	numParticles = 1; 
	testSet = [0]*21; 
	acts = [0]*21; 
	for i in range(0,len(testSet)):
		testSet[i] = GM(i,0.001,1).sample(numParticles); 
		[distSort1,dists1,eta1] = shepardsInterpolation(V,testSet[i],retDist=True); 
		grid[0][i] = V[dists1.index(distSort1[0])][2]; 

	axarr[1].set_xlim([0,20]); 
	axarr[1].imshow(grid,extent=[0,20,0,1],cmap='inferno');
	axarr[1].legend(['Left','Right','Stay']); 
	axarr[1].set_title('Comparison to MDP policy implementation'); 

	plt.show(); 

if __name__ == "__main__":

	#testParticleFilter(); 
	#testMCPOMDP(); 

	V = np.load('MCPolicy1.npy').tolist(); 
	#displayPolicy(V); 

	#compareToMDP(V); 

	checkBoth(V); 

