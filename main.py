import numpy as np
import operator
import matplotlib.pyplot as plt

# loads reward data
reward = np.loadtxt("./reward.csv",delimiter=",")

Lx = 10
Ly = 10
s_goal = Lx*Ly
a = [0,1,2,3] # right, left, down ,up
gamma = 0.5  # discount rate
kk=0

def state2coor(s):
	return (int(np.floor(1.0*s/Lx)), np.mod(s,Lx))
def coor2state(i,j):
	return i*Lx + j

class Agent:
	def __init__(self):
		self.s=0
		self.last_s = self.s
		self.last_a = 0
		self.t=0		
		self.Q = np.zeros([100,4])

	def get_epsilon(self):
		return 0.2
	def get_alpha(self):
		"""
		calculates learning rate
		"""
		return 0.5
	def zero(self):
		self.s=0
		self.last_s = self.s
		self.last_a = 0
		self.t=0

	def act(self):
		global kk
		if self.s != s_goal:
			while self.s==self.last_s:

				a_index, Q_max = max( enumerate(self.Q[self.s,:]), key=operator.itemgetter(1) )
				rdm = np.random.random()
				epsilon = self.get_epsilon()
				if rdm <=epsilon:
					tmp=[0,1,2,3]
					tmp.remove(a_index)
					print(tmp)
					a = np.random.choice( tmp )
				else:
					a = a_index
				self.last_a = a

				i,j=state2coor(self.s)
				print(i,j)
				kk+=1
				if a==0 and j<Lx-1:
					j += 1
				elif a==1 and j>0:
					j -= 1
				elif a==2 and i<Ly-1:
					i += 1
				elif a==3 and i>0:
					i -= 1
				self.s = coor2state(i, j)
				print(self.s)

			self.t += 1 
			return True
		else:
			return False

	def obtain_reward(self, s, a):
		global reward
		return reward[s,a]

	def updateQ(self):
		alpha = self.get_alpha()
		r = self.obtain_reward(self.last_s, self.last_a)
		self.Q[self.last_s, self.last_a] = \
			self.Q[self.last_s, self.last_a] + alpha*(
				r + gamma*max(self.Q[self.s,:]) - self.Q[self.last_s, self.last_a] )
		self.last_s = self.s

def main():
	global kk
	agent=Agent()
	while agent.act():
		agent.updateQ()
	print(kk)
if __name__=="__main__":
	main()






