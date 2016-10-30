from . import nn

import math
import pylab as pl
import numpy as np

class BPLayer(nn.Layer):

	def __init__(self, preSize, mySize, nnFun):
		nn.Layer.__init__(self, preSize, mySize, nnFun)

	def getDiagDerivativeMatrix(self):
		
		diags = None

		if self.nnFun.derInY == True:
			diags = np.copy(self.outputs)	
			nn.MatrixOP.apply(diags, self.nnFun.derivative)
		else:
			diags = np.copy(self.neurons)
			nn.MatrixOP.apply(diags, self.nnFun.derivative)

		if diags.shape == (1, 1):
			return diags
		
		return np.diag(np.squeeze(np.asarray(diags)))
		

class BackPropagationNeuralNetwork(nn.Network):
	
	def __init__(self, alpha, sizeOfLayers, activationFunctions):
		nn.Network.__init__(self, alpha, sizeOfLayers, activationFunctions)

		self.layers = [BPLayer(1, sizeOfLayers[0], activationFunctions[0])]
		for i in range(1, self.nnDepth):
			self.layers.append(BPLayer(sizeOfLayers[i - 1], sizeOfLayers[i], activationFunctions[i]))

	def forward(self, inputVector):

		# set first layer equal to input <-> neuronList[0]
		self.layers[0].outputs = inputVector

		for i in range(1, self.nnDepth):
			curLayer = self.layers[i]
			preLayer = self.layers[i - 1]
			curLayer.receiveSignal(preLayer.outputs)
			# if i == 1:
			# print("layer %d outputs" % (i-1))
			# print(preLayer.outputs)
			# print("layer %d weight" % (i))
			# print(curLayer.weight)
			# print("layer %d bias" % (i))
			# print(curLayer.bias)


			# print("layer %d neurons" % (i))
			# print(curLayer.neurons)
			# print("layer %d outputs" % (i))
			# print(curLayer.outputs)

		# print("=================================")
		# print("output")
		# print(self.layers[self.outputIndex].outputs)
		# input("next>>")

			

		return self.layers[self.outputIndex].outputs

	def dspforward(self, inputVector):

		# set first layer equal to input <-> neuronList[0]
		self.layers[0].outputs = inputVector

		for i in range(1, self.nnDepth):
			curLayer = self.layers[i]
			preLayer = self.layers[i - 1]
			curLayer.dspreceiveSignal(preLayer.outputs)
			# if i == 1:
			print("preLayer.outputs\n", preLayer.outputs)
			print("curLayer.neurons\n", curLayer.neurons)
			# 	input("next>>")

			

		return self.layers[self.outputIndex].outputs

	def backPropagation(self, errVector):

		# print("errVector")
		# print(errVector)

		outputLayer = self.layers[self.outputIndex]

		sensitivity = -2 * np.dot(outputLayer.getDiagDerivativeMatrix(), errVector)

		for i in range(0, self.outputIndex)[::-1]:

			curLayer = self.layers[i]
			nxtLayer = self.layers[i + 1]

			nxtSensitivity = np.copy(sensitivity);

			sensitivity = np.dot(
				np.dot(
					curLayer.getDiagDerivativeMatrix(), np.transpose(nxtLayer.weight))
				, nxtSensitivity)

			# print("adjust weight %d" % (i+1))
			# print(-1 * self.alpha * np.dot(nxtSensitivity, np.transpose(curLayer.outputs)))
			# print("adjust bias %d" % (i+1))
			# print(-1 * self.alpha * nxtSensitivity)


			nxtLayer.learnWeight(-1 * self.alpha * np.dot(nxtSensitivity, np.transpose(curLayer.outputs)))
			nxtLayer.learnBias(-1 * self.alpha * nxtSensitivity)


def init(alpha, sizeOfLayers, activationFunctions):
	return BackPropagationNeuralNetwork(alpha, sizeOfLayers, activationFunctions)