'''
######################################################

File: FIB_Example.py
Author: Luke Burks
Date: April 2017

Implements a Fast Informed Bound POMDP
on the discrete hallway problem

Written for a guest lecture for Nisar Ahmeds 
Aerospace Algorithms for Autonomy class


######################################################
'''
from __future__ import division
from copy import deepcopy;
import matplotlib.pyplot as plt; 
import numpy as np; 
import matplotlib.animation as animation
import matplotlib.image as mgimg

class FIBSolver():


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


	def solve(self):
		self.solveMDP(); 
		self.findQ(); 

	def solveMDP(self):
		
		self.V = [min(min(self.model.R))]*self.model.N; 
		W = [np.random.random()]*self.model.N; 

		while(not self.listComp(self.V,W)):
			W = deepcopy(self.V); 
			for i in range(0,self.model.N):
				self.V[i] = self.model.discount * max(self.model.R[a][i] + sum(self.V[j]*self.model.px[a][i][j] for j in range(0,self.model.N)) for a in range(0,self.model.acts)); 
	

	def findQ(self):
		self.Q = np.zeros(shape=(self.model.acts,self.model.N)).tolist(); 

		for a in range(0,self.model.acts):
			for i in range(0,self.model.N):
				self.Q[a][i] += self.model.R[a][i];
				for o in range(0,self.model.acts):
					self.Q[a][i] += sum(self.model.pz[o][j]*self.V[j]*self.model.px[a][i][j] for j in range(0,self.model.N));

	def getAction(self,bel):
		y = [0]*self.model.acts; 
		y[0] = sum(bel[i]*self.Q[0][i] for i in range(0,self.model.N)); 
		y[1] = sum(bel[i]*self.Q[1][i] for i in range(0,self.model.N)); 
		y[2] = sum(bel[i]*self.Q[2][i] for i in range(0,self.model.N)); 

		act = np.argmax(y); 
		return act; 



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


def compareToMDP(ans):
	grid = np.ones(shape=(1,ans.model.N)); 

	plt.figure();
	plt.scatter(-5,.5,c='k'); 
	plt.scatter(-5,.5,c='r');
	plt.scatter(-5,.5,c='y');

	for i in range(0,ans.model.N):
		bel = [0]*ans.model.N;  
		bel[i] = 1; 
		grid[0][i] = ans.getAction(bel);
	plt.xlim([0,20]); 
	plt.imshow(grid,extent=[0,ans.model.N-1,0,1],cmap='inferno');
	plt.legend(['Left','Right','Stay']); 
	plt.title('Comparison to MDP policy implementation'); 

	plt.show(); 

def simulate(ans):
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
		act = ans.getAction(b);
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

def seeAlphas(ans):
	x = [i for i in range(0,ans.model.N)]; 

	for i in range(0,len(ans.Q)):
		if(i == 0.0):
			col = 'k'; 
		elif(i == 1.0):
			col = 'r'; 
		else:
			col = 'y'; 
		plt.plot(x,ans.Q[i],c=col);
	plt.show(); 

def checkBoth(ans):
	fig,axarr = plt.subplots(2,sharex=True); 

	x = [i for i in range(0,ans.model.N)]; 

	for i in range(0,len(ans.Q)):
		if(i == 0.0):
			col = 'k'; 
		elif(i == 1.0):
			col = 'r'; 
		else:
			col = 'y'; 
		axarr[0].plot(x,ans.Q[i],c=col,linewidth=5);
	axarr[0].set_title('FIB Alpha Vectors'); 

	grid = np.ones(shape=(1,ans.model.N)); 

	
	axarr[1].scatter(-5,.5,c='k'); 
	axarr[1].scatter(-5,.5,c='r');
	axarr[1].scatter(-5,.5,c='y');

	for i in range(0,ans.model.N):
		bel = [0]*ans.model.N;  
		bel[i] = 1; 
		grid[0][i] = ans.getAction(bel);
	axarr[1].set_xlim([0,20]); 
	axarr[1].imshow(grid,extent=[0,ans.model.N-1,0,1],cmap='inferno');
	axarr[1].legend(['Left','Right','Stay']); 
	axarr[1].set_title('Comparison to MDP policy implementation'); 

	plt.show(); 


def animate(ans):

	B = []; 
	X = []; 
	#uniform intial belief
	b = [1/ans.model.N]*ans.model.N; 
	x = 4; 

	B.append(b); 
	X.append(x); 

	allState = [i for i in range(0,ans.model.N)]; 
	allObs = [i for i in range(0,ans.model.obs)]; 

	fig = plt.figure(); 
	plt.plot(allState,b,linewidth=6);
	plt.scatter(x,0.2,c='#ffa500',s=250);  
	plt.ylim([0,.3]); 
	plt.xlim([0,20]); 
	plt.pause(.1); 
	for count in range(0,100):
		act = ans.getAction(b);
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

		B.append(b); 
		X.append(x); 
		if(x == 13 and act == 2):
			break; 





	#Show Results
	fig,ax = plt.subplots(1,sharex=True); 
	x = [i for i in range(0,21)]; 
	for i in range(0,len(B)):
		ax.cla()
		ax.plot(x,B[i],linewidth=5);
		ax.set_title('time step='+str(i))
		ax.set_xlabel('position (m)')
		ax.set_ylabel('belief')
		ax.scatter(X[i],0.3,c='g',s = 200); 
		ax.set_xlim([0,20]); 
		ax.set_ylim([0,.5]); 
		
		#grab temp images
		fig.savefig('../tmp/img'+str(i)+".png",bbox_inches='tight',pad_inches=0)
		plt.pause(.1)
	

	#Animate Results
	fig,ax=plt.subplots()
	images=[]
	for k in range(0,len(B)):
		fname='../tmp/img%d.png' %k
		img=mgimg.imread(fname)
		imgplot=plt.imshow(img)
		plt.axis('off')
		images.append([imgplot])
	ani=animation.ArtistAnimation(fig,images,interval=20)


if __name__ == "__main__":
	ans = FIBSolver(); 
	ans.solve(); 

	#seeAlphas(ans); 

	#compareToMDP(ans); 

	#checkBoth(ans); 
	animate(ans); 
	#simulate(ans); 