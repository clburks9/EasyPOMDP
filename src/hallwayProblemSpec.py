'''
######################################################

File: hallwayProblemSpec.py
Author: Luke Burks
Date: April 2017

Specs out portas hallway problem in discrete space
for use by multiple solvers
Written for a guest lecture for Nisar Ahmeds 
Aerospace Algorithms for Autonomy class


1D problem, from 0 to 20, with goal at 13; 
Increments of 1

3 actions: left,right,stay
stay is deterministic
left and right have some chance to not actually move

3 observations: left,right,near, with noise
bad sensor, for any left or right .6 chance correct
.2 chance each other
for near, .6 chance correct around outlet, with .2
chance each other. and .8 chance correct at outlet,
with .1 chance each other


Rewards: 100 for stay at 2
		 -1 for everything left or right
		 -5 for stay not at 2
		 -10 for left or right at the edge


######################################################
'''

import numpy as np; 
import matplotlib.pyplot as plt;

class ModelSpec:

	def __init__(self):
		self.discount = .9; 
		self.N = 21; 
		self.acts = 3; 
		self.obs = 5; 

		self.buildTransition(); 
		self.buildObservations(); 
		self.buildRewards(); 

		

	def buildTransition(self):
		self.px = np.zeros(shape=(3,self.N,self.N)).tolist(); 

		for i in range(0,self.N):
			for j in range(0,self.N):
				if(j == i):
					self.px[2][i][j] = 1; 
				elif(j==i-1):
					self.px[0][i][j] = .8; 
					self.px[0][i][i] = .2;
				elif(j==i+1):
					self.px[1][i][j] = .8; 
					self.px[1][i][i] = .2; 

		#edge cases
		self.px[0][0][0] = 1; 
		self.px[1][self.N-1][self.N-1] = 1;


	def buildObservations(self):
		#3 and 4 are far left and far right

		self.pz = np.zeros(shape = (5,self.N)).tolist(); 
		self.pz[0][0] = .05; 
		self.pz[1][0] = .05; 
		self.pz[2][0] = .05; 
		self.pz[4][0] = .05; 
		self.pz[3][0] = .8; 

		self.pz[0][1] = .05; 
		self.pz[1][1] = .05; 
		self.pz[2][1] = .05; 
		self.pz[4][1] = .05; 
		self.pz[3][1] = .8; 
		for i in range(2,12):
			self.pz[0][i] = .8; 
			self.pz[1][i] = .1; 
			self.pz[2][i] = .1; 
		self.pz[0][12] = .15; 
		self.pz[1][12] = .15; 
		self.pz[2][12] = .7; 
		self.pz[0][13] = .025; 
		self.pz[1][13] = .025; 
		self.pz[2][13] = .95; 
		self.pz[0][14] = .15; 
		self.pz[1][14] = .15; 
		self.pz[2][14] = .7; 
		for i in range(15,self.N-2):
			self.pz[0][i] = .1; 
			self.pz[1][i] = .8; 
			self.pz[2][i] = .1; 

		self.pz[0][self.N-1] = .05; 
		self.pz[1][self.N-1] = .05; 
		self.pz[2][self.N-1] = .05; 
		self.pz[3][self.N-1] = .05; 
		self.pz[4][self.N-1] = .8; 

		self.pz[0][self.N-2] = .05; 
		self.pz[1][self.N-2] = .05; 
		self.pz[2][self.N-2] = .05; 
		self.pz[3][self.N-2] = .05; 
		self.pz[4][self.N-2] = .8; 

	def buildRewards(self):
		self.R = np.zeros(shape=(3,self.N)).tolist(); 

		for i in range(0,self.N):
			self.R[0][i] = -1; 
			self.R[1][i] = -1; 
			self.R[2][i] = -5; 
		self.R[2][13] = 100; 
		self.R[0][0] = -10; 
		self.R[1][self.N-1] = -10; 

		


def checkReward(m):
	x = [i for i in range(0,m.N)]; 
	plt.plot(x,m.R[0],c='r',linewidth=5);
	plt.plot(x,m.R[1],c='b',linewidth=5); 
	plt.plot(x,m.R[2],c='g',linewidth=5);  
	plt.legend(['Left','Right','Stay']); 
	plt.title('Hallway Problem Reward Function'); 
	plt.show()

def checkObs(m):
	x = [i for i in range(0,m.N)]; 
	plt.plot(x,m.pz[0],c='r',linewidth=5);
	plt.plot(x,m.pz[1],c='b',linewidth=5); 
	plt.plot(x,m.pz[2],c='g',linewidth=5); 
	plt.plot(x,m.pz[3],c='k',linewidth=5); 
	plt.plot(x,m.pz[4],c='y',linewidth=5); 
	plt.legend(['Left','Right','Near','Far Left','Far Right']); 
	plt.title('Hallway Problem Observation Model'); 
	plt.show()

def checkTransition(m):
	x0 = [[0 for i in range(0,m.N)] for j in range(0,m.N)]; 
	x1 = [[0 for i in range(0,m.N)] for j in range(0,m.N)]; 
	x2 = [[0 for i in range(0,m.N)] for j in range(0,m.N)]; 
	
	x, y = np.mgrid[0:m.N:1, 0:m.N:1]


	for i in range(0,m.N):
		for j in range(0,m.N):
			x0[i][j] = m.px[0][i][j]; 
			x1[i][j] = m.px[1][i][j]; 
			x2[i][j] = m.px[2][i][j]; 


	fig,axarr = plt.subplots(3); 
	axarr[0].contourf(x,y,x0); 
	axarr[1].contourf(x,y,x1); 
	axarr[2].contourf(x,y,x2); 
	plt.show(); 
	

if __name__ == "__main__":
	m = ModelSpec(); 
	
	#checkReward(m);
	checkObs(m); 
	#checkTransition(m); 

