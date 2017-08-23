'''
######################################################

File: PBVI_Example.py
Author: Luke Burks
Date: April 2017

Implements a Point-Based Value iteration POMDP
on the discrete hallway problem

Written for a guest lecture for Nisar Ahmeds 
Aerospace Algorithms for Autonomy class


######################################################
'''
from __future__ import division
from copy import deepcopy;
import matplotlib.pyplot as plt; 
import numpy as np; 
import random

class PBVISolver():


	def __init__(self):
		modelModule = __import__('hallwayProblemSpec', globals(), locals(), ['ModelSpec'],0); 
		modelClass = modelModule.ModelSpec;
		self.model = modelClass(); 


	def listComp(self,a,b): 
		if(len(a) != len(b)):
			return False; 

		for i in range(0,len(a)):
			if(a[i] != b[i]):
				return False; 

		return True; 


	def solve(self,numIter = 100):
		self.gatherBeliefs(); 
		self.solvePBVI(numIter); 

	def gatherBeliefs(self):
		
		self.B = []; 

		bInit = np.zeros(shape=(self.model.N,self.model.N)).tolist(); 
		for i in range(0,self.model.N):
			bInit[i][i] = 1; 
			suma=1; 
			for j in range(0,self.model.N):
				if(j!=i):
					bInit[i][j] = np.random.random()/5;
					suma+=bInit[i][j]; 
			for j in range(0,self.model.N):
				bInit[i][j] = bInit[i][j]/suma; 

		self.B = bInit; 



	def precomputeAls(self):
		als1 = np.zeros(shape=(len(self.Gamma),self.model.acts,self.model.obs,self.model.N)).tolist(); 

		for g in range(0,len(self.Gamma)):
			for a in range(0,self.model.acts):
				for o in range(0,self.model.obs):
					for s in range(0,self.model.N):
						for sprime in range(0,self.model.N):
							als1[g][a][o][s] += self.model.pz[o][sprime]*self.model.px[a][s][sprime]*self.Gamma[g][sprime]; 
		return als1


	def dotProduct(self,a,b):
		suma = 0; 
		for i in range(0,len(a)):
			suma+=a[i]*b[i]; 
		return suma; 


	def newBackup(self,b,als1):
		bestVal = -10000000000; 
		bestAct = 0; 
		bestAlpha = [];

		for a in range(0,self.model.acts):
			suma = [0]*self.model.N; 
			for o in range(0,self.model.obs):
				best = als1[np.argmax([self.dotProduct(b,als1[j][a][o]) for j in range(0,len(als1))])][a][o];
				for i in range(0,self.model.N):
					suma[i]+= self.model.R[a][i] + self.model.discount*best[i];

			tmp = self.dotProduct(b,suma);

			if(tmp > bestVal):
				bestAct = a; 
				bestAlpha= deepcopy(suma); 
				bestVal = tmp;

		
		bestAlpha.append(bestAct); 
		return bestAlpha; 

	def prune(self):

		bestGam = [0]*self.model.N; 

		for i in range(0,self.model.N):
			tmpVal = -1000; 
			tmpInd = 0; 

			for g in range(0,len(self.Gamma)):
				if(self.Gamma[g][i] > tmpVal):
					tmpVal = self.Gamma[g][i]; 
					tmpInd = g; 
			bestGam[i] = tmpInd; 
		toDel = []; 
		for i in range(0,len(self.Gamma)):
			if(i not in bestGam):
				toDel.append(self.Gamma[i]);

		for g in toDel:
			self.Gamma.remove(g);  

	def argDot(self,Gamma,b):

		tmpVal = -10000; 
		tmpInd = 0; 
		for i in range(0,len(Gamma)):
			tmp = self.dotProduct(b,Gamma[i]); 
			if(tmp>tmpVal):
				tmpVal = tmp; 
				tmpInd = i; 
		return Gamma[tmpInd]; 

	def solvePBVI(self,counter = 10):

		self.Gamma = [[min(min(self.model.R))/(1-self.model.discount)]*self.model.N]

	
		for count in range(0,counter):
			print("Iteration: " + str(count+1)); 

			GammaNew = []; 

			preAls = self.precomputeAls(); 

			for b in self.B:

				#al = self.backup(b,preAls);
				al = self.newBackup(b,preAls);


				#make sure the alpha doesn't already exist					
				addFlag = False;
				for g in GammaNew: 
					if(not self.listComp(al,g)):
						addFlag = True;
				if(addFlag):
					GammaNew.append(al); 
				
				GammaNew.append(al); 


			self.Gamma = deepcopy(GammaNew);
			for i in range(0,len(self.Gamma)):
				self.Gamma[i] = self.Gamma[i][:self.model.N+1]

			self.prune();
			#print(self.Gamma); 


			if(count < 5):
				Bnew = []; 
				for b in self.B:
					Bnew.append(b);
					allState = [i for i in range(0,ans.model.N)]; 
					allObs = [i for i in range(0,ans.model.obs)]; 	
					x = np.random.choice(allState,p=b);   
					act = self.getAction(b,self.Gamma); 
					ztrial = [0]*ans.model.obs; 
					for i in range(0,ans.model.obs):
						ztrial[i] = ans.model.pz[i][x]; 
					z = np.random.choice(allObs,p=ztrial)

					Bnew.append(self.beliefUpdate(b,act,z)); 
				self.B = deepcopy(Bnew); 
			

		f = open('PBVI_Policy.npy','w'); 
		np.save(f,self.Gamma); 



	def getAction(self,b,Gamma):
		bestVal = -100000; 
		bestInd = 0; 

		for j in range(0,len(Gamma)):
			tmp = self.dotProduct(b,Gamma[j]); 
			if(tmp>bestVal):
				bestVal = tmp; 
				bestInd = j; 
		return int(Gamma[bestInd][-1]); 


	def beliefUpdate(self,bel,a,o):
		belBar = [0]*self.model.N; 
		belNew = [0]*self.model.N; 
		suma = 0; 
		for i in range(0,self.model.N):
			belBar[i] = sum(self.model.px[a][j][i]*bel[j] for j in range(0,self.model.N)); 
			belNew[i] = self.model.pz[o][i]*belBar[i];
			suma+=belNew[i];  
		#normalize
		for i in range(0,self.model.N):
			belNew[i] = belNew[i]/suma; 

		return belNew; 

def seeAlphas(ans):
	Gamma = np.load('PBVI_Policy.npy').tolist();

	#Gamma = ans.B; 

	x = [i for i in range(0,ans.model.N)]; 

	for g in Gamma:
		if(g[-1] == 0.0):
			col = 'k'; 
		elif(g[-1] == 1.0):
			col = 'r'; 
		else:
			col = 'y'; 
		plt.plot(x,g[:ans.model.N],c=col);
	plt.show(); 

def compareToMDP(ans):
	grid = np.ones(shape=(1,ans.model.N)); 

	Gamma = np.load('PBVI_Policy.npy').tolist(); 

	for g in Gamma:
		print(g[-1]);


	plt.figure();
	plt.scatter(-5,.5,c='k'); 
	plt.scatter(-5,.5,c='r');
	plt.scatter(-5,.5,c='y');

	for i in range(0,ans.model.N):
		bel = [0]*ans.model.N;  
		bel[i] = 1; 
		grid[0][i] = ans.getAction(bel,Gamma);
	plt.xlim([0,20]); 
	plt.imshow(grid,extent=[0,ans.model.N-1,0,1],cmap='inferno');
	plt.legend(['Left','Right','Stay']); 
	plt.title('Comparison to MDP policy implementation'); 

	plt.show(); 

def checkBoth(ans):
	fig,axarr = plt.subplots(2,sharex=True); 

	Gamma = np.load('../policies/PBVI_Policy.npy').tolist();

	#Gamma = ans.B; 

	x = [i for i in range(0,ans.model.N)]; 

	for g in Gamma:
		if(g[-1] == 0.0):
			col = 'k'; 
		elif(g[-1] == 1.0):
			col = 'r'; 
		else:
			col = 'y'; 
		axarr[0].plot(x,g[:ans.model.N],c=col,linewidth=3);
	axarr[0].set_title('PBVI Alpha Vectors');  

	grid = np.ones(shape=(1,ans.model.N)); 


	axarr[1].scatter(-5,.5,c='k'); 
	axarr[1].scatter(-5,.5,c='r');
	axarr[1].scatter(-5,.5,c='y');

	for i in range(0,ans.model.N):
		bel = [0]*ans.model.N;  
		bel[i] = 1; 
		grid[0][i] = ans.getAction(bel,Gamma);
	axarr[1].set_xlim([0,20]); 
	axarr[1].imshow(grid,extent=[0,ans.model.N-1,0,1],cmap='inferno');
	axarr[1].legend(['Left','Right','Stay']); 
	axarr[1].set_title('Comparison to MDP policy implementation'); 

	plt.show(); 


def simulate(ans):
	#uniform intial belief

	Gamma = np.load('PBVI_Policy.npy').tolist(); 

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
		act = ans.getAction(b,Gamma);
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
	ans = PBVISolver(); 
	#ans.solve(15); 

	#seeAlphas(ans); 

	#compareToMDP(ans); 

	checkBoth(ans); 

	#simulate(ans); 
