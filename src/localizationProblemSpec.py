'''
######################################################

File: localizationProblemSpec.py
Author: Luke Burks
Date: April 2017

Specs out a simple 2D localization problem
Written for a guest lecture for Nisar Ahmeds 
Aerospace Algorithms for Autonomy class


2D problem, from 0 to 9, 0 to 9
Goal at (2,4); 

5 actions: left,right,up,down,stay
stay is deterministic
left and right , up and down, have some chance to not 
actually move and to go sideways

5 observations: quadrants,near, with noise
decent sensor

obstacles at (4,4), (7,2) (7,7) and (1,6); 

Rewards: 100 for stay at (2,4)
		 -1 for everything
		 -5 for stay not at (2,4)
		 -10 for hitting an obstacle


######################################################
'''

import numpy as np; 
import matplotlib.pyplot as plt;

class ModelSpec:

	def __init__(self):
		self.discount = .9; 
		self.N = 100; 
		self.acts = 5; 
		self.obs = 5; 

		print('Building Transition'); 
		self.buildTransition();
		print('Building Observations');  
		self.buildObservations(); 
		print('Building Rewards'); 
		self.buildRewards(); 

	def convertToGridCords(self,i):
		y = i//10; 
		x = i%10; 
		return x,y; 

	def buildTransition(self):
		self.px = np.zeros(shape=(self.acts,self.N,self.N)).tolist(); 

		#left,right,up,down,stay

		for i in range(0,self.N):
			[x1,y1] = self.convertToGridCords(i); 
			for j in range(0,self.N):
				[x2,y2] = self.convertToGridCords(j);
				if(x1 == x2 and y1==y2):
					self.px[4][i][j] = 1; 
					self.px[0][i][j] = 0;
					self.px[1][i][j] = 0; 
					self.px[2][i][j] = 0; 
					self.px[3][i][j] = 0;  
				elif(y1 == y2 and x1 == x2+1):
					self.px[4][i][j] = .05; 
					self.px[0][i][j] = .8;
					self.px[1][i][j] = .05; 
					self.px[2][i][j] = .05; 
					self.px[3][i][j] = .05;
				elif(y1 == y2 and x1 == x2-1):
					self.px[4][i][j] = .05; 
					self.px[1][i][j] = .8;
					self.px[0][i][j] = .05; 
					self.px[2][i][j] = .05; 
					self.px[3][i][j] = .05;
				elif(y1 == y2-1 and x1 == x2):
					self.px[4][i][j] = .05; 
					self.px[2][i][j] = .8;
					self.px[1][i][j] = .05; 
					self.px[0][i][j] = .05; 
					self.px[3][i][j] = .05;
				elif(y1 == y2+1 and x1 == x2):
					self.px[4][i][j] = .05; 
					self.px[3][i][j] = .8;
					self.px[1][i][j] = .05; 
					self.px[2][i][j] = .05; 
					self.px[0][i][j] = .05;

		#normalize
		for a in range(0,self.acts):
			for i in range(0,self.N):
				suma = 0; 
				for j in range(0,self.N):
					suma+=self.px[a][i][j]; 
				for j in range(0,self.N):
					self.px[a][i][j] = self.px[a][i][j]/suma; 
		


	def buildObservations(self):
		#upper left, upper right, lower left, lower right, near

		self.pz = np.ones(shape = (self.obs,self.N)).tolist();

		for i in range(0,self.N):
			[x,y] = self.convertToGridCords(i); 

			#upper left, 0,9 to 4,4
			if(x<4 and y> 4):
				self.pz[0][i] = 3; 
			elif(x>=4 and y>4):
				self.pz[1][i] = 3; 
			elif(x<4 and y<=4):
				self.pz[2][i] = 3; 
			elif(x>=4 and y<=4):
				self.pz[3][i] = 3; 
			if(x==2 and y==4):
				self.pz[4][i] = 100; 
		#normalize:
		for o in range(0,self.obs):
			suma=0; 
			for i in range(0,self.N):
				suma+=self.pz[o][i]; 
			for i in range(0,self.N):
				self.pz[o][i] = self.pz[o][i]/suma; 



	def buildRewards(self):
		self.R = np.zeros(shape=(5,self.N)).tolist(); 

		#obstacles at (4,4), (7,2) (7,7) and (2,5); 
		obstacles = [[4,4],[7,2],[7,7],[2,7]]; 
		negRew = -10000; 
		for i in range(0,self.N):
			[x,y] = self.convertToGridCords(i); 
			if([x-1,y] in obstacles):
				self.R[0][i] = negRew; 
			elif([x+1,y] in obstacles):
				self.R[1][i] = negRew; 
			elif([x,y-1] in obstacles):
				self.R[2][i] = negRew; 
			elif([x,y+1] in obstacles):
				self.R[3][i] = negRew; 
			elif([x,y] == [2,4]):
				self.R[4][i] = 100; 
			else:
				for j in range(0,self.acts):
					self.R[j][i] = -1; 


	def convertToGrid(self,a):

		b = np.zeros(shape=(10,10)).tolist();  
		for i in range(0,100):
			b[i//10][i%10]=a[i]; 
		return b; 

def checkReward(m):
	'''
	x = [i for i in range(0,m.N)]; 
	plt.plot(x,m.R[0],c='r');
	plt.plot(x,m.R[1],c='b'); 
	plt.plot(x,m.R[2],c='g');  
	plt.legend(['Left','Right','Stay']); 
	plt.show()
	'''
	fig,axarr = plt.subplots(m.acts); 
	for a in range(0,m.acts):
		axarr[a].contourf(m.convertToGrid(m.R[a]),cmap = 'inferno')
	plt.show(); 

def checkObs(m):

	fig,axarr = plt.subplots(m.obs); 
	for o in range(0,m.obs):
		axarr[o].contourf(m.convertToGrid(m.pz[o]),cmap = 'inferno')
	plt.show(); 

def checkTransition(m):
	x0 = [[0 for i in range(0,m.N)] for j in range(0,m.N)]; 
	x1 = [[0 for i in range(0,m.N)] for j in range(0,m.N)]; 
	x2 = [[0 for i in range(0,m.N)] for j in range(0,m.N)]; 
	x3 = [[0 for i in range(0,m.N)] for j in range(0,m.N)]; 
	x4 = [[0 for i in range(0,m.N)] for j in range(0,m.N)]; 

	x, y = np.mgrid[0:m.N:1, 0:m.N:1]


	for i in range(0,m.N):
		for j in range(0,m.N):
			x0[i][j] = m.px[0][i][j]; 
			x1[i][j] = m.px[1][i][j]; 
			x2[i][j] = m.px[2][i][j];
			x3[i][j] = m.px[1][i][j]; 
			x4[i][j] = m.px[2][i][j]; 



	fig,axarr = plt.subplots(5); 
	axarr[0].contourf(x,y,x0); 
	axarr[1].contourf(x,y,x1); 
	axarr[2].contourf(x,y,x2); 
	axarr[3].contourf(x,y,x1); 
	axarr[4].contourf(x,y,x2);
	plt.show(); 
	

if __name__ == "__main__":
	m = ModelSpec(); 

	checkReward(m);
	checkObs(m); 
	checkTransition(m); 

