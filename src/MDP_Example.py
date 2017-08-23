'''
######################################################

File: MDP_Example.py
Author: Luke Burks
Date: April 2017

Implements an MDP on the discrete hallway problem

Written for a guest lecture for Nisar Ahmeds 
Aerospace Algorithms for Autonomy class


######################################################
'''

from copy import deepcopy;
import matplotlib.pyplot as plt; 
import numpy as np; 


class MDPSolver():


	def __init__(self,d=1):
		if(d==1):
			modelModule = __import__('hallwayProblemSpec', globals(), locals(), ['ModelSpec'],0); 
		elif(d==2):
			modelModule = __import__('localizationProblemSpec', globals(), locals(), ['ModelSpec'],0); 
		modelClass = modelModule.ModelSpec;
		self.model = modelClass();  

	def listComp(self,a,b,d=1): 
		if(len(a) != len(b)):
			return False; 
		for i in range(0,len(a)):
			if(a[i] != b[i]):
				return False;

		return True; 


	def solve(self,d=1):
		
		print('Finding Value Function'); 


		self.V = [min(min(self.model.R))]*self.model.N; 
		W = [np.random.random()]*self.model.N; 
	
		while(not self.listComp(self.V,W,d)):
			W = deepcopy(self.V); 
			for i in range(0,self.model.N):
				self.V[i] = self.model.discount * max(self.model.R[a][i] + sum(self.V[j]*self.model.px[a][i][j] for j in range(0,self.model.N)) for a in range(0,self.model.acts)); 
	
	def getAction(self,x):	
		y = [0]*self.model.acts; 
		for i in range(0,self.model.acts):
			y[i] = self.model.R[i][x] + sum(self.V[j]*self.model.px[i][x][j] for j in range(0,self.model.N)); 
		act = np.argmax(y); 
		return act; 

def checkValue(ans):
	plt.plot(ans.V); 
	plt.title('Value Function'); 
	plt.show(); 


def checkPolicy(ans):
	colorGrid = ['g','r','b']; 
	grid = np.ones(shape=(1,ans.model.N)); 
	for i in range(1,ans.model.N):
		grid[0][i] = ans.getAction(i);
	plt.imshow(grid,extent=[0,ans.model.N-1,0,1],cmap='inferno');
	plt.title('Policy Implementation'); 
	plt.show(); 

def checkBoth(ans):
	fig,axarr = plt.subplots(2,sharex=True);
	axarr[0].plot(ans.V,linewidth=5); 
	axarr[0].legend(['Value Function']);

	colorGrid = ['g','r','b']; 
	grid = np.ones(shape=(1,ans.model.N)); 
	for i in range(1,ans.model.N):
		grid[0][i] = ans.getAction(i);
	axarr[1].scatter(-5,.5,c='k'); 
	axarr[1].scatter(-5,.5,c='r');
	axarr[1].scatter(-5,.5,c='y');

	axarr[1].imshow(grid,extent=[0,ans.model.N-1,0,1],cmap = 'inferno');
	axarr[1].set_xlim([0,20]); 
	axarr[1].set_title('Policy Implementation'); 
	axarr[1].legend(['Left','Right','Stay']); 
	plt.suptitle('Hallway MDP'); 

	plt.show();

def convertToGridCords(i):
	y = i//10; 
	x = i%10; 
	return x,y;  
def convertToGrid(a):

	b = np.zeros(shape=(10,10)).tolist();  
	for i in range(0,100):
		b[i//10][i%10]=a[i]; 
	return b; 

def checkBoth2D(ans):
	Vtilde = convertToGrid(ans.V); 

	grid = np.ones(shape=(10,10)); 
	for i in range(0,ans.model.N):
		[x,y] = convertToGridCords(i)
		grid[y][x] = ans.getAction(i);

	arrowGridU = np.zeros(shape=(10,10)); 
	arrowGridV = np.zeros(shape=(10,10)); 

	for y in range(0,10):
		for x in range(0,10):
			if(grid[y][x] == 0.0):
				arrowGridU[x][y] = -1; 
			elif(grid[y][x] == 1.0):
				arrowGridU[x][y] = 1;
			elif(grid[y][x] == 2.0):
				arrowGridV[x][y] = 1; 
			elif(grid[y][x] == 3.0):
				arrowGridV[x][y] = -1;

	X,Y = np.mgrid[0:10:1, 0:10:1]


	U = np.cos(X); 
	V = np.sin(Y); 

	obstacles = [[4,4],[7,2],[7,7],[2,5]]; 
	obsX = [4,7,7,2]; 
	obsY = [4,2,7,5]; 
	plt.contourf(Vtilde,cmap = 'viridis');
	plt.scatter(obsX,obsY,c='r',s=150,marker='8'); 
	plt.scatter(2,4,c='c',s=250,marker='*')
	plt.quiver(X,Y,arrowGridU,arrowGridV,color='k'); 
	plt.title('MDP Value Function and Policy'); 
	plt.show(); 



if __name__ == "__main__":
	ans = MDPSolver(d=1); 
	ans.solve(d=1);  

	#checkValue(ans); 
	#checkPolicy(ans); 
	checkBoth(ans);  
	#checkBoth2D(ans); 




