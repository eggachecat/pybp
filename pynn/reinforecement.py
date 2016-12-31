import numpy as np
import copy



class ACE():
	def __init__(self, boxNum, greek):

		self.mvx = np.zeros((boxNum, 1))
		self.v = np.zeros((boxNum, 1))

		self.predict = 0
		self.prePredict = 0


		# discount factor for future reinf
		self.gamma = greek["gamma"]

		self.beta = greek["beta"]

		# trace decay rate
		self.eta = greek["eta"]

	def update(self):
		pass

	def updatePredict(self, x):

		self.prePredict = copy.deepcopy(self.predict)
		self.predict = np.dot(np.transpose(self.v), x)

	def updateVvalue(self, reinf):

		self.v += self.beta * (reinf + self.gamma * self.predict - self.prePredict)

	def updateMovingAverageX(self, x):

		self.mvx = self.eta * self.mvx + (1 - self.eta) * x

	def reinforce(self, x, reinf):

		self.updatePredict(x)
		self.updateVvalue(reinf)
		self.updateMovingAverageX(x)


		# print("mvx", self.mvx)
		# print("vvalue", self.v)

		return reinf + self.gamma * self.predict - self.prePredict

class ReinforecementNeuralNetwork():

	def __init__(self, svs, statesPartiton, failureStatesPartitions, actionSet, greek, beginState = None, beginAction = None):


		# state variables
		self.svs = svs
		self.failureStatesPartitions = failureStatesPartitions
		self.statesPartiton = statesPartiton

		self.iniBoxBases()
		self.iniState(beginState)

		# action
		self.actionSet = actionSet		
		self.iniAction(beginAction)

		self.qval = np.zeros((self.totalBoxes, len(actionSet)), dtype = float)
		# self.vval = np.zeros((self.totalBoxes, len(actionSet)), dtype = float)

		# learning rate
		self.alpha = greek["alpha"]

		# learing rate for ACE
		self.beta = greek["beta"]

		# discount factor for future reinf
		self.gamma = greek["gamma"]

		# magnitude of noise added to choice 
		self.delta = greek["delta"]

		self.boxVector = np.zeros((self.totalBoxes, 1), dtype = float)

		self.ace = ACE(self.totalBoxes, greek)

		# self.reinf = reinf


	def Qvalue(self, state, action):

		rowIndex = self.getBox(state)
		colIndex = self.__actionColumnMap[action]

		return self.qval[rowIndex, colIndex]

	def updateQvalue(self, state, action, newVal):


		rowIndex = self.getBox(state)
		colIndex = self.__actionColumnMap[action]

		
		self.qval[rowIndex, colIndex] = newVal

	# choose action with max Q value give state
	def chooseMaxQvalue(self, state):
		maxQ = -np.inf
		for action in self.actionSet:
			trail = self.Qvalue(state, action)
			if maxQ < trail:
				maxQ = trail

		return maxQ

	def chooseAction(self, state, withNoise = False):

		maxQ = -np.inf
		optimalAction = self.actionSet[0]
		for action in self.actionSet:
			trail = self.Qvalue(state, action) + int(withNoise) * self.delta * np.random.random_sample()
			if maxQ < trail:
				maxQ = trail
				optimalAction = action

		return optimalAction

	def iniAction(self, beginAction):

		if not beginAction:
			self.beginAction = self.actionSet[0]
		

		self.action = self.beginAction
		self.preAction = self.action


		self.__actionColumnMap = dict()
		for i in range(0, len(self.actionSet)):
			self.__actionColumnMap[self.actionSet[i]] = i



	def iniState(self, beginState):

		if not beginState:
			self.beginState = dict()
			for sv in self.svs:
				self.beginState[sv] = 0.0
		else:
			self.beginState = beginState

		self.state = self.beginState
		self.box = self.getBox(self.state)

		self.preState = self.state

	def iniBoxBases(self):
		self.bases = dict()
		self.totalBoxes = 1
		for sv in self.svs:
			self.bases[sv] = self.totalBoxes
			self.totalBoxes *= len(self.statesPartiton[sv])


	def setState(self, state):

		self.preState = copy.deepcopy(self.state)
		self.preAction = copy.deepcopy(self.action)

		for sv in self.svs:
			self.state[sv] = state[sv]

		# new box-vector
		self.boxVector.fill(0)

		self.box = self.getBox(self.state)
		if not self.box < 0:
			self.boxVector[self.box, 0] = 1


	def getBoxValue(self, partitions, state):

		_boxValue = 0

		for sv in self.svs:

			if not sv in partitions:
				continue

			judgers = partitions[sv]
			base = self.bases[sv]

			for box in sorted(judgers):
				judger = judgers[box]
				if judger(state[sv]):
					_boxValue += base * int(box)
					break;

		return _boxValue - 1

	def getStateBox(self, state):
		return self.getBoxValue(self.statesPartiton, state)

	def isFailed(self, state):
		return self.getBoxValue(self.failureStatesPartitions, state) < -1

	def getBox(self, state):

		if self.isFailed(state):
			return -1
		else:
			return self.getStateBox(state)

	def get_action(self, reward = 0):

		# self.preState = self.state
		# self.preAction = self.action

		self.box = self.getBox(self.state)

		if self.getBox(self.preState) > 0:
			if self.box < 0:
				predicted_value = 0
			else:
				predicted_value = self.chooseMaxQvalue(self.state)
		
			# (1-alpha)*q + alpha*(r + gamma * max(q))
			newQvalue = (1 - self.alpha) * self.Qvalue(self.preState, self.preAction) + self.alpha * (reward + self.gamma * predicted_value)

			# print("not failed: set[%d, %d] = %f" % (self.getBox(self.preState), self.preAction, newQvalue))
			self.updateQvalue(self.preState, self.preAction, newQvalue)
		else:
			pass


		self.action = self.chooseAction(self.state, withNoise = True)

		return self.action

	def failed_update(self, punish = -1):

		predicted_value = 0

		# (1-alpha)*q + alpha*(r + gamma * max(q))
		newQvalue = (1 - self.alpha) * self.Qvalue(self.preState, self.preAction) + self.alpha * (punish + self.gamma * predicted_value)

		# print("failed: set[%d, %d] = %f" % (self.getBox(self.preState), self.preAction, newQvalue))
		self.updateQvalue(self.preState, self.preAction, newQvalue)

		# if self.Q_VALUES(self.state, 1) + (rand * BETA) <= self.Q_VALUES(self.state, 2):
		#     cur_action = 2;  
		# else:
		# 	cur_action = 1;

	# if state failed
	def ifFailed(self, punish = -1):

		# print("current state:", self.state)

		self.box = self.getBox(self.state)

		# print(self.preState, self.state)

		# print(self.box)

		# current state failed
		if self.box < 0:
			self.failed_update(punish)
			return True

		return False


	def ace_ifFailed(self, punish = -1):

		# print("current state:", self.state)

		self.box = self.getBox(self.state)

		# print(self.preState, self.state)

		# print(self.box)

		# current state failed
		if self.box < 0:
			self.ace_failed_update(punish)
			return True

		return False


	# the state value function no-longer be the max(Q)
	# insread, it follows the equation : predict_value = V(S_t+1) = 
	def ace_get_action(self, reward = 0):

		# self.preState = self.state
		# self.preAction = self.action

		self.box = self.getBox(self.state)

		if self.getBox(self.preState) > 0:
			if self.box < 0:
				predicted_value = 0
			else:
				predicted_value = self.chooseMaxQvalue(self.state)

			# reward = self.ace.reinforce(x, -1)

			reward = self.ace.reinforce(self.boxVector, 0)

			# if not reward == 0:
			# 	print("get_action reward: ", reward)

		
			# (1-alpha)*q + alpha*(r + gamma * max(q))
			newQvalue = (1 - self.alpha) * self.Qvalue(self.preState, self.preAction) + self.alpha * (reward + self.gamma * predicted_value)

			# print("not failed: set[%d, %d] = %f" % (self.getBox(self.preState), self.preAction, newQvalue))
			self.updateQvalue(self.preState, self.preAction, newQvalue)
		else:
			pass


		self.action = self.chooseAction(self.state, withNoise = True)

		return self.action

	def ace_failed_update(self, punish = -1):

		predicted_value = 0


		reward = self.ace.reinforce(self.boxVector, -1)

		# if not reward == 0:
		# 	print("ace_failed_update reward: ", reward)


		newQvalue = (1 - self.alpha) * self.Qvalue(self.preState, self.preAction) + self.alpha * (punish + self.gamma * predicted_value)

		# print("failed: set[%d, %d] = %f" % (self.getBox(self.preState), self.preAction, newQvalue))
		self.updateQvalue(self.preState, self.preAction, newQvalue)





