from . import nn

import math
import pylab as pl
import numpy as np


import itertools



class SOMLayer(nn.Layer):

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

		# print(vector_r)
		# print(vector_c)

		max = float("-inf")
		min = float("inf")

		maxTuple = (0, 0, max)
		minTuple = (0, 0, min)

		distanceMatrix = np.zeros((rowNum, colNum), dtype=np.float)
		for r in range(0, rowNum):
			for c in range(0, colNum):
				difference = vector_r[r] - vector_c[c]
				distance = np.sum(np.square(difference))
				distanceMatrix[r, c] = np.absolute(distance)

				if(distance > max):
					max = distance
					maxTuple = (r, c, max)
				else :
					if (distance < min):
						min = distance
						minTuple = (r, c, min)

		print("\nThe distance matrix is\n")
		print(distanceMatrix)

		return maxTuple, minTuple
 



	# two class
	def train(self, dataSet, trainLayerIndex):

		# print("================================================================================================")
		patterns = dict()
		# trainLayerIndex = 1

		curlayer = self.layers[trainLayerIndex]

		preLayer = self.layers[trainLayerIndex - 1]
		cur_inputs_T = np.transpose(preLayer.outputs)


		print("current weight is:")
		print(curlayer.weight)
		print("current bias is:")
		print(curlayer.bias, "\n")


		patterns_input_dict = dict()


		print("IN THE forward SECTION:============================================================")
		for pair in dataSet:
			key = pair["category"]
			if key not in patterns:
				patterns[key] = []

			output = self.forward(np.transpose(pair["input"]), trainLayerIndex)

			print("INPUT: ", pair["input"])
			print("OUTPUT: ", np.transpose(output), "\n")



			patterns_input_dict[output.tostring()] = np.transpose(preLayer.outputs)
			# print(output)
			patterns[key].append(output)



		print("IN THE training SECTION====================================================================")

		# combination of catogries!!
		for keys in list(itertools.combinations_with_replacement(list(patterns.keys()), 2)):
			# print(keys, patterns[keys[0]], patterns[keys[1]])

			print("for key: ", keys[0], "-", keys[1])

			distanceTuple = self.createDistanceMatrix(patterns[keys[0]], patterns[keys[1]])


			the_same_category = (keys[0] == keys[1])
			# print(keys[0], keys[1])

			# same class
			if the_same_category:
				## same cat
				## min the max distance
				targetIndices = distanceTuple[0]
				multiplier = self.attRate

			else:
				# otherwise
				targetIndices = distanceTuple[1]
				multiplier = -1 * self.repRate


			print("\nThe critical vectors are: \n")


			# the two bad-asses
			vector_p = patterns[keys[0]][targetIndices[0]]
			vector_q = patterns[keys[1]][targetIndices[1]]

			print(np.transpose(vector_p))
			print("\nand\n")
			print(np.transpose(vector_q))


			# if the_same_category:
			# 	print("same class:")
			# 	print(tmp_dict[vector_p.tostring()])
			# 	print( tmp_dict[vector_q.tostring()] )
			# else:
			# 	print("not same class:")
			# 	print(tmp_dict[vector_p.tostring()])
			# 	print(tmp_dict[vector_q.tostring()] )


			# input("enter.....")




			######################### assume in Y form for now - begin #########################
			diag_p = np.copy(vector_p)
			nn.MatrixOP.apply(diag_p, curlayer.nnFun.derivative)

			diag_q = np.copy(vector_q)
			nn.MatrixOP.apply(diag_q, curlayer.nnFun.derivative)
			######################### assume in Y form for now - end #########################

			## this is the matrix whose diag are the derivatives of neurons
			diag_q = np.diag(np.squeeze(np.asarray(diag_q)))
			diag_p = np.diag(np.squeeze(np.asarray(diag_p)))

			print("\nTheir diag-directive Matrices are: \n")
			print(diag_p)
			print("\nand\n")			
			print(diag_q)

			# sensitivity = np.dot(diag_p - diag_q, vector_p - vector_q)

			print("\nTheir difference Matrices are: \n")
			print(vector_p - vector_q)

			sensitivity_p = np.dot(diag_p, vector_p - vector_q)
			sensitivity_q = np.dot(diag_q, vector_p - vector_q)

			print("\nTheir sensitivity Matrices are: \n")
			print(sensitivity_p)
			print("\nand\n")
			print(sensitivity_q)

			print("\nTheir sensitivity In-flow are: \n")
			print(patterns_input_dict[vector_p.tostring()])
			print("\nand\n")
			print(patterns_input_dict[vector_q.tostring()])


			# print(sensitivity_p, "\n", patterns_input_dict[vector_p.tostring()])
			# print(sensitivity_q, "\n", patterns_input_dict[vector_q.tostring()])

			adjust_p = np.dot(sensitivity_p, patterns_input_dict[vector_p.tostring()])
			adjust_q = np.dot(sensitivity_q, patterns_input_dict[vector_q.tostring()])

			print("\nTheir jusr Matrices are: \n")
			print(adjust_p)
			print("\nand\n")
			print(adjust_q)

			updateWeight = -1 * multiplier * (adjust_p - adjust_q)

			updateBias = -1 * multiplier * (sensitivity_p - sensitivity_q)

			print("\nSO The weight updated value is:\n")
			print(updateWeight)
			print("\nSO The bias updated value is:\n")
			print(updateBias)


		# 	print(updateWeight)
		# 	print(updateBias)

		# 	print("+++++++++++++++++++++++++")

			curlayer.learnWeight(updateWeight)
			curlayer.learnBias(updateBias)

		print("after this itr, the weight is now:")
		print(curlayer.weight)
		print("the bias is now:")
		print(curlayer.bias)
		print("================================================================================================")




def init(attRate, repRate, sizeOfLayers, activationFunctions):
	return SelfOrganizingMapNeuralNetwork(attRate, repRate, sizeOfLayers, activationFunctions)