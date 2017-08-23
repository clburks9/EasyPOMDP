'''
######################################################

File: catProblemSpec.py
Author: Luke Burks
Date: April 2017

Specs out an outdoor cat feeder problem
for use by multiple solvers
Written for a final project in Nisar Ahmed's
Aerospace Algorithms for Autonomy class


3 states: S={Cat1,Cat2,Other}

3 actions: A={Feed_Large,Feed_Small,No_Feed}

3 observations: O = {Cat1,Cat2,Other}


######################################################
'''

import numpy as np; 
import matplotlib.pyplot as plt;

class ModelSpec:

	def __init__(self):
		self.discount = .9; 
		self.N = 3; 
		self.acts = 3; 
		self.obs = 3; 

		self.buildTransition(); 
		self.buildObservations(); 
		self.buildRewards(); 

		

	def buildTransition(self):
		self.px = np.zeros(shape=(self.acts,self.N,self.N)).tolist(); 

		self.px[0][0][0] = .1; 
		self.px[0][0][1] = .7;
		self.px[0][0][2] = .2;

		self.px[0][1][0] = .4; 
		self.px[0][1][1] = .1;
		self.px[0][1][2] = .5;

		self.px[0][2][0] = .1; 
		self.px[0][2][1] = .1;
		self.px[0][2][2] = .8;


		self.px[1][0][0] = .4; 
		self.px[1][0][1] = .4;
		self.px[1][0][2] = .2;

		self.px[1][1][0] = .7; 
		self.px[1][1][1] = .1;
		self.px[1][1][2] = .2;

		self.px[1][2][0] = .2; 
		self.px[1][2][1] = .2;
		self.px[1][2][2] = .6;
		

		self.px[2][0][0] = .8; 
		self.px[2][0][1] = .1;
		self.px[2][0][2] = .1;

		self.px[2][1][0] = .1; 
		self.px[2][1][1] = .8;
		self.px[2][1][2] = .1;

		self.px[2][2][0] = .4; 
		self.px[2][2][1] = .4;
		self.px[2][2][2] = .2;


	def buildObservations(self):

		self.pz = np.zeros(shape = (self.obs,self.N)).tolist(); 
		
		self.pz[0][0] = .7; 
		self.pz[0][1] = 0; 
		self.pz[0][2] = .25; 

		self.pz[1][0] = 0; 
		self.pz[1][1] = .8182; 
		self.pz[1][2] = .25; 
		
		self.pz[2][0] = .5; 
		self.pz[2][1] = .1818;  
		self.pz[2][2] = .5; 
		


	def buildRewards(self):
		self.R = np.zeros(shape=(self.acts,self.N)).tolist(); 

		self.R[0][0] = 10; 
		self.R[0][1] = 2; 
		self.R[0][2] = -20; 

		self.R[1][0] = 2; 
		self.R[1][1] = 10; 
		self.R[1][2] = -10; 

		self.R[2][0] = -5; 
		self.R[2][1] = -5; 
		self.R[2][2] = 5; 


def checkReward(m):
	x = [i for i in range(0,m.N)]; 
	plt.plot(x,m.R[0],c='r',linewidth=5);
	plt.plot(x,m.R[1],c='b',linewidth=5); 
	plt.plot(x,m.R[2],c='g',linewidth=5);  
	plt.legend(['Feed_Large','Feed_Small','No_Feed']); 
	plt.title('Cat Problem Reward Function'); 
	plt.show()

def checkObs(m):
	x = [i for i in range(0,m.N)]; 
	plt.plot(x,m.pz[0],c='r',linewidth=5);
	plt.plot(x,m.pz[1],c='b',linewidth=5); 
	plt.plot(x,m.pz[2],c='g',linewidth=5); 

	plt.legend(['Cat1','Cat2','Other']); 
	plt.title('Cat Problem Observation Model'); 
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
	#checkObs(m); 
	#checkTransition(m); 

