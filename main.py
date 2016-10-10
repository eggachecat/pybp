import math
import pylab as pl
import numpy as np

def sigmoid(x):
	return 1 / (1 + np.exp(-1 * x))

def purelin(x):
	return x




class NNFunction():
	def __init__(self, activate, derInY, derivative):
		self.activate = activate

		# derivative in y or x expression
		self.derInY = derInY 
		self.derivative = derivative

class NNMatrix():
	def __init__(self):
		pass

	@staticmethod
	def apply(matrix, fn):
		for x in np.nditer(matrix, op_flags=['readwrite']):
			x[...] = fn(x)

	@staticmethod
	def pipe(matrix, fn):
		newMatrix = np.copy(matrix)
		for x in np.nditer(newMatrix, op_flags=['readwrite']):
			x[...] = fn(x)

		return newMatrix

class NeuralLayer:
	def __init__(self, preSize, mySize, nnFun):
		self.size = mySize
		self.neurons = np.zeros((mySize, 1), dtype=float) 
		self.outputs = np.zeros((mySize, 1), dtype=float)
		self.weight = np.ones((mySize, preSize), dtype=float)
		self.bias = np.zeros((mySize, 1), dtype=float)
		self.nnFun = nnFun

	def receiveSignal(self, signals):
		self.neurons = np.dot(self.weight, signals) + self.bias
		self.outputs = NNMatrix.pipe(self.neurons, self.nnFun.activate)

	def outputSignal(self):
		return self.outputs

	def getDiagDerivativeMatrix(self):
		
		diags = None

		if self.nnFun.derInY == True:
			diags = np.copy(self.outputs)
			NNMatrix.apply(diags, self.nnFun.derivative)
		else:
			diags = np.copy(self.neurons)
			NNMatrix.apply(diags, self.nnFun.derivative)

		if diags.shape == (1, 1):
			return diags
		
		return np.diag(np.squeeze(np.asarray(diags)))
		

	def learnWeight(self, adjust):
		self.weight = self.weight + adjust;

	def learnBias(self, bias):
		self.bias = self.bias + bias;

	def setNeurons(self, neurons):
		self.neurons = neurons

	def setOutputs(self, outputs):
		self.outputs = outputs

	def setWeight(self, weight):
		self.weight = weight

	def setBias(self, bias):
		self.bias = bias



class NeuralNetwork:
	"""NeuralNetwork"""
	def __init__(self, alpha, sizeOfLayers, activationFunctions):

		""" sizeOfLayers 
			#0 -> input size

			#last -> output size
		"""

		self.alpha = alpha

		self.layerNum = len(sizeOfLayers)
		self.outputIndex = self.layerNum - 1

		self.layers = [NeuralLayer(1, sizeOfLayers[0], activationFunctions[0])]

		for i in range(1, self.layerNum):
			self.layers.append(NeuralLayer(sizeOfLayers[i - 1], sizeOfLayers[i], activationFunctions[i]))
		
	
	def forward(self, inputVector):

		# set first layer equal to input <-> neuronList[0]
		self.layers[0].setOutputs(inputVector)
		print(self.layers[0].outputs)

		for i in range(1, self.layerNum):
			curLayer = self.layers[i]
			preLayer = self.layers[i - 1]
			curLayer.receiveSignal(preLayer.outputSignal())
			

		return self.layers[self.outputIndex].outputSignal()

	def backPropagation(self, errVector):

		outputLayer = self.layers[self.outputIndex]

		sensitivity = -2 * np.dot(outputLayer.getDiagDerivativeMatrix(), errVector)

		for i in range(0, self.outputIndex)[::-1]:

			preLayer = self.layers[i - 1]
			curLayer = self.layers[i]
			nxtLayer = self.layers[i + 1]

			currSensitivity = np.copy(sensitivity);

			sensitivity = np.dot(
				np.dot(
					curLayer.getDiagDerivativeMatrix(), np.transpose(nxtLayer.weight))
				, currSensitivity)


			nxtLayer.learnWeight(-1 * self.alpha * np.dot(currSensitivity, np.transpose(curLayer.outputs)))
			nxtLayer.learnBias(-1 * self.alpha * currSensitivity)

		

		

afs = [NNFunction(lambda x: x, False, lambda x: 1), 
	   NNFunction(sigmoid, True, lambda y: y * (1-y)), 
	   NNFunction(lambda x: x, False, lambda x: 1)]	
NN = NeuralNetwork(0.1, [1, 2, 1], afs)

NN.layers[1].setWeight(np.mat([
	[-0.27], 
	[-0.41]
]))

NN.layers[1].setBias(np.mat([
	[-0.48], 
	[-0.13]
]))

NN.layers[2].setWeight(np.mat([
	[0.09, -0.17]
]))

NN.layers[2].setBias(np.mat([
	[0.48]
]))

output = NN.forward(np.mat([
	[1.]
]))


teacher = np.mat([
	[1.7071]
])

NN.backPropagation(teacher - output)

print(NN.layers[2].weight)
print(NN.layers[2].bias)
print(NN.layers[1].weight)
print(NN.layers[1].bias)