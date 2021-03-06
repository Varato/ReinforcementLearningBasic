import numpy as np
import operator
import matplotlib.pyplot as plt

# loads reward data
reward = np.loadtxt("./reward.csv",delimiter=",")

Lx = 10
Ly = 10
s_goal = Lx*Ly-1
# a = [0,1,2,3] # up, right, down ,left
gamma = 0.9  # discount rate

def state2coor(s):
	return (np.mod(s,Ly), int(np.floor(1.0*s/Ly)) )
def coor2state(i,j):
	return j*Ly + i

class Agent:
	def __init__(self):
		self.s=0
		self.last_s = self.s
		self.last_a = 0
		self.t=1		
		self.Q = np.zeros([100,4])

	def get_epsilon(self, learning=True):
		if learning:
			return (1 + 5*np.log(self.t))/self.t
			# return 0.2
		else:
			return 0
	def get_alpha(self):
		"""
		calculates learning rate
		"""
		return (1 + np.log(self.t))/self.t
		# return 0.5
	def zero(self):
		self.s=0
		self.last_s = self.s
		self.last_a = 0
		self.t=1

	def act(self, learning=True):

		if self.s != s_goal:
			a_index, Q_max = max( enumerate(self.Q[self.s,:]), key=operator.itemgetter(1) )
			rdm = np.random.random()
			epsilon = self.get_epsilon(learning)
			if rdm <=epsilon:
				tmp=[0,1,2,3]
				tmp.remove(a_index)
				a = np.random.choice( tmp )
			else:
				a = a_index
			self.last_a = a

			i,j=state2coor(self.s)
			if a==1 and j<Lx-1:
				j += 1
			elif a==3 and j>0:
				j -= 1
			elif a==2 and i<Ly-1:
				i += 1
			elif a==0 and i>0:
				i -= 1
			self.s = coor2state(i, j)
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

def main(N=3000, max_cost=1000):
	agent=Agent()
	for i in range(N):
		cost=0
		print("current trial: {}".format(i+1))
		agent.zero()
		while agent.act():
			cost += 1
			agent.updateQ()
			if cost >= max_cost:
				break
		print("cost: {}".format(cost))

	# record the optimal path
	s_record = [[0,0]]
	action_record = []
	agent.zero()
	while agent.act(learning=False):
		i,j=state2coor(agent.s)
		s_record.append([i, j])
		action_record.append(agent.last_a)


	s_record = np.array(s_record)
	action_record = np.array(action_record)
	np.savetxt("s_record.csv", s_record)
	np.savetxt("action_record.csv", action_record)

	print(s_record)
	
	


if __name__=="__main__":
	main()






