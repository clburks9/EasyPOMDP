'''
######################################################

File: QMDP_Example.py
Author: Luke Burks
Date: April 2017

Implements a QMDP on the discrete hallway problem

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


class QMDPSolver():


	def __init__(self,d=1):
		if(d==1):
			modelModule = __import__('hallwayProblemSpec', globals(), locals(), ['ModelSpec'],0); 
		elif(d==2):
			modelModule = __import__('localizationProblemSpec', globals(), locals(), ['ModelSpec'],0); 
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
				self.Q[a][i] = self.model.R[a][i] + sum(self.V[j]*self.model.px[a][i][j] for j in range(0,self.model.N)); 

	def getAction(self,bel):
		y = [0]*self.model.acts; 
		for j in range(0,self.model.acts):
			y[j] = sum(bel[i]*self.Q[j][i] for i in range(0,self.model.N)); 
		

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

def convertToGridCords(i):
	y = i//10; 
	x = i%10; 
	return x,y;  
def convertToGrid(a):

	b = np.zeros(shape=(10,10)).tolist();  
	for i in range(0,100):
		b[i//10][i%10]=a[i]; 
	return b; 


def compare2D(ans):
	Vtilde = convertToGrid(ans.V); 

	grid = np.ones(shape=(10,10)); 
	for i in range(0,ans.model.N):
		bel = [0]*ans.model.N;  
		bel[i] = 1; 
		[x,y] = convertToGridCords(i)
		grid[y][x] = ans.getAction(bel);

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
	axarr[0].set_title('QMDP Alpha Vectors'); 

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
	x = 1; 

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
		fig.savefig('./tmp/img'+str(i)+".png",bbox_inches='tight',pad_inches=0)
		plt.pause(.1)
	

	#Animate Results
	fig,ax=plt.subplots()
	images=[]
	for k in range(0,len(B)):
		fname='./tmp/img%d.png' %k
		img=mgimg.imread(fname)
		imgplot=plt.imshow(img)
		plt.axis('off')
		images.append([imgplot])
	ani=animation.ArtistAnimation(fig,images,interval=20)
	ani.save("QMDP_animation.gif",fps=2,writer='animation.writer')

if __name__ == "__main__":
	ans = QMDPSolver(d=1); 
	ans.solve(); 

	#seeAlphas(ans); 
	#compareToMDP(ans);

	#checkBoth(ans); 

	animate(ans); 

	#simulate(ans); 
	#compare2D(ans); 