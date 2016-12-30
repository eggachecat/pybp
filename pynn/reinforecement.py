import numpy as np
import copy



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

		self.qval = np.zeros((1 + self.totalBoxes, len(actionSet)), dtype = float)

		# learning rate
		self.alpha = greek["alpha"]

		# learing rate for ACE
		self.beta = greek["beta"]

		# discount factor for future reinf
		self.gamma = greek["gamma"]

		# magnitude of noise added to choice 
		self.delta = greek["delta"]

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

		return _boxValue

	def getStateBox(self, state):
		return self.getBoxValue(self.statesPartiton, state)

	def isFailed(self, state):
		return self.getBoxValue(self.failureStatesPartitions, state) < 0

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






