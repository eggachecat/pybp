from . import nn

import math
import pylab as pl
import numpy as np


import itertools



class SOMLayer(nn.Layer):

	def __init__(self, preSize, mySize, nnFun):
		nn.Layer.__init__(self, preSize, mySize, nnFun)



class SelfOrganizingMapNeuralNetwork(nn.Network):


	def __init__(self, attRate, repRate, sizeOfLayers, activationFunctions):
		nn.Network.__init__(self, sizeOfLayers, activationFunctions)
		self.attRate = attRate
		self.repRate = repRate

		self.layers = [SOMLayer(1, sizeOfLayers[0], activationFunctions[0])]
		for i in range(1, self.nnDepth):
			self.layers.append(SOMLayer(sizeOfLayers[i - 1], sizeOfLayers[i], activationFunctions[i]))

	def forward(self, inputVector, maxDepth):

		# set first layer equal to input <-> neuronList[0]
		self.layers[0].outputs = inputVector

		for i in range(1, maxDepth + 1):
			curLayer = self.layers[i]
			preLayer = self.layers[i - 1]
			curLayer.receiveSignal(preLayer.outputs)

		return self.layers[maxDepth].outputs

	def createDistanceMatrix(self, vector_r, vector_c):


		rowNum = len(vector_r)
		colNum = len(vector_c)

		max = float("-inf")
		min = float("inf")

		maxIndeices = (0, 0)
		minIndecies = (0, 0)

		distanceMatrix = np.zeros((rowNum, colNum), dtype=np.float)
		for r in range(0, rowNum):
			for c in range(0, colNum):
				difference = vector_r[r] - vector_c[c]
				distance = np.sum(np.square(difference))
				distanceMatrix[r, c] = np.absolute(distance)

				if(distance > max):
					max = distance
					maxIndeices = (r, c)
				else :
					if (distance < min):
						min = distance
						minIndecies = (r, c)

		return maxIndeices, minIndecies
 

	# two class
	def train(self, dataSet):

		patterns = dict()
		trainLayerIndex = 1

		for pair in dataSet:
			key = pair["category"]
			if key not in patterns:
				patterns[key] = []

			output = self.forward(np.transpose(pair["input"]), trainLayerIndex)
			# print(output)
			patterns[key].append(output)

		# combination of catogries!!
		for keys in list(itertools.combinations_with_replacement(list(patterns.keys()), 2)):
			# print(keys, patterns[keys[0]], patterns[keys[1]])


			indicesTuple = self.createDistanceMatrix(patterns[keys[0]], patterns[keys[1]])

			if keys[0] == keys[1]:
				## same cat
				## min the max distance
				targetIndices = indicesTuple[0]

			else:
				# otherwise
				targetIndices = indicesTuple[1]

			print(targetIndices)

def init(attRate, repRate, sizeOfLayers, activationFunctions):
	return SelfOrganizingMapNeuralNetwork(attRate, repRate, sizeOfLayers, activationFunctions)