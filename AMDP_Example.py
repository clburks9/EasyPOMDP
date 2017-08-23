'''
######################################################

File: AMDP_Example.py
Author: Luke Burks
Date: April 2017

Implements an AMDP on the discrete hallway problem

Written for a guest lecture for Nisar Ahmeds 
Aerospace Algorithms for Autonomy class



######################################################
'''


from __future__ import division
from copy import deepcopy;
import matplotlib.pyplot as plt; 
import numpy as np; 
from scipy.stats import norm; 
import time; 

class AMDPSolver():


	def __init__(self):
		modelModule = __import__('hallwayProblemSpec', globals(), locals(), ['ModelSpec'],0); 
		modelClass = modelModule.ModelSpec;
		self.model = modelClass();
		self.numVaris = 10; 

	def beliefUpdate(self,bel,a,o):
		belBar = [0]*self.model.N; 
		belNew = [0]*self.model.N; 
		suma = 0; 
		for i in range(0,self.model.N):
			belBar[i] = sum(self.model.px[a][i][j]*bel[j] for j in range(0,self.model.N)); 
			belNew[i] = self.model.pz[o][i]*belBar[i];
			suma+=belNew[i];  
		#normalize
		for i in range(0,self.model.N):
			belNew[i] = belNew[i]/suma; 

		return belNew; 

	def normalize(self,a):
		suma = 0; 
		b=[0]*len(a); 
		for i in range(0,len(a)):
			suma+=a[i]
		for i in range(0,len(a)):
			b[i] = a[i]/suma; 
		return b; 

	def calEntropy(self,b):
		ent = 0; 
		for i in range(0,len(b)):
			ent += -b[i]*np.log(b[i]); 
		return ent; 

	def calStats(self,b,allVars):
		mean = np.argmax(b); 
		#var = np.var(b);
		ent = self.calEntropy(b); 
		ent = min(allVars, key=lambda x:abs(x-ent))

		varInd = allVars.index(ent); 
		return [mean,varInd]; 

	def listComp(self,a,b): 
		if(len(a) != len(b)):
			return False; 

		for i in range(0,len(a)):
			for j in range(0,len(a[i])):
				if(a[i][j] != b[i][j]):
					return False; 

		return True; 



	def solve(self):

		numVaris = self.numVaris; 
		allState = [i for i in range(0,ans.model.N)]; 
		allObs = [i for i in range(0,ans.model.obs)]; 

		maxEnt = 3.04452243772; 
		self.allVars = np.arange(0.01,maxEnt,maxEnt/numVaris).tolist(); 
		allVars = self.allVars; 

		
		loopCount = 10; 

		 

		augB = np.zeros(shape=(self.model.N,numVaris)).tolist();
		Phat = np.zeros(shape=(self.model.acts,self.model.N,numVaris,self.model.N,numVaris)).tolist(); 
		Rhat = np.zeros(shape=(self.model.acts,self.model.N,numVaris)).tolist(); 
		for i in range(0,self.model.N):
			for j in range(0,numVaris):
				for a in range(0,self.model.acts):
					for c in range(0,loopCount):
						mean = i;  
						var = allVars[j]; 
						b=[0]*self.model.N; 
						for h in range(0,self.model.N):
							b[h] = norm.pdf(h,mean,var); 
						b = self.normalize(b); 
						x = np.random.choice(allState,p=b); 
						xprime = np.random.choice(allState,p=self.model.px[a][x]); 

						ztrial = [0]*self.model.obs; 
						for h in range(0,self.model.obs):
							ztrial[h] = self.model.pz[h][x]; 
						z = np.random.choice(allObs,p=ztrial)

						bprime = self.beliefUpdate(b,a,z); 
						[iprime,jprime] = self.calStats(bprime,allVars); 
						
						Phat[a][i][j][iprime][jprime] += 1/loopCount; 
						Rhat[a][i][j] += self.model.R[a][x]/loopCount;

		print('State space reduced. Solving Policy'); 



		Vhat = (np.ones(shape=(self.model.N,numVaris))*min(min(self.model.R))).tolist(); 
		W = [[-10000 for i in range(0,numVaris)] for i in range(0,self.model.N)]; 
		while(not self.listComp(Vhat,W)):
			W = deepcopy(Vhat); 
			for i in range(0,self.model.N):
				for j in range(0,numVaris):
					acts = [0]*self.model.acts; 
					for a in range(0,self.model.acts):
						for k in range(0,self.model.N):
							for l in range(0,numVaris):
								acts[a] += Vhat[k][l]*Phat[a][i][j][k][l]; 
						acts[a] += Rhat[a][i][j]; 
					Vhat[i][j] = self.model.discount*max(acts);  

		f = open('AMDP_Policy2.npy','w'); 
		sal = [Vhat,Phat,Rhat]; 
		np.save(f,sal); 

	def getAction(self,Vhat,Phat,Rhat,b):
		allVars = [i/2 for i in range(0,self.numVaris)]; 
		bBar = self.calStats(b,allVars); 

		acts = [0]*self.model.acts; 
		for a in range(0,len(acts)):
			for i in range(0,self.model.N):
				for j in range(0,self.numVaris):
					acts[a] += Vhat[i][j]*Phat[a][bBar[0]][bBar[1]][i][j]; 
			acts[a] += Rhat[a][bBar[0]][bBar[1]]; 
		return np.argmax(acts); 


def loadPolicy(fileName):
	[Vhat,Phat,Rhat] = np.load(fileName).tolist(); 
	return [Vhat,Phat,Rhat]; 

def compareToMDP(ans):


	allVars = [i/2 for i in range(0,ans.numVaris)]; 
	[Vhat,Phat,Rhat] = loadPolicy('AMDP_Policy2.npy'); 
 

	grid = np.ones(shape=(1,ans.model.N)); 

	plt.figure();
	plt.scatter(-5,.5,c='k'); 
	plt.scatter(-5,.5,c='r');
	plt.scatter(-5,.5,c='y');

	for i in range(0,ans.model.N):
		bel = [0]*ans.model.N;  
		bel[i] = 1; 
		grid[0][i] = ans.getAction(Vhat,Phat,Rhat,bel);
	plt.xlim([0,20]); 
	plt.imshow(grid,extent=[0,ans.model.N-1,0,1],cmap='inferno');
	plt.legend(['Left','Right','Stay']); 
	plt.title('Comparison to MDP policy implementation'); 

	plt.show(); 

def simulate(ans):

	[Vhat,Phat,Rhat] = loadPolicy('AMDP_Policy2.npy'); 

	#uniform intial belief
	b = [1/ans.model.N]*ans.model.N; 
	x = 4; 
	allState = [i for i in range(0,ans.model.N)]; 
	allObs = [i for i in range(0,ans.model.obs)]; 

	fig = plt.figure(); 
	plt.plot(allState,b,linewidth=6);
	plt.scatter(x,0.2,c='#ffa500',s=250);  
	plt.ylim([0,.3]); 
	plt.xlim([0,20]); 
	plt.pause(.1); 
	for count in range(0,100):

		act = ans.getAction(Vhat,Phat,Rhat,b);
		if(x == 0):
			act = 1; 
		elif(x ==ans.model.N-1):
			act=0; 

		x = np.random.choice(allState,p=ans.model.px[act][x]); 

		ztrial = [0]*ans.model.obs; 
		for i in range(0,ans.model.obs):
			ztrial[i] = ans.model.pz[i][x]; 
		z = np.random.choice(allObs,p=ztrial)

		b = ans.beliefUpdate(b,act,z);
		#print(np.argmax(b),x,act,z);  

		col = '#ffa500'; 
		if(act == 2 and x!= 13):
			col = 'r'; 
		plt.clf(); 
		plt.plot(allState,b,linewidth=6);
		plt.scatter(x,0.2,c=col,s=250);  
		plt.ylim([0,.3]); 
		plt.xlim([0,20]); 
		plt.pause(.1); 

		if(x == 13 and act == 2):
			break; 
	if(x == 13 and act == 2):
		plt.clf(); 
		plt.plot(allState,b,linewidth=6);
		plt.scatter(x,0.2,c='g',s=250);  
		plt.ylim([0,.3]); 
		plt.xlim([0,20]); 
		plt.pause(.1); 
		plt.show(); 

if __name__ == "__main__":
	ans = AMDPSolver(); 
	#ans.solve();


	compareToMDP(ans); 
	#simulate(ans); 