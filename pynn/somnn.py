from pynn import nn
from pynn import nnaf

import math
import pylab as pl
import numpy as np


import itertools

import logging


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


	def __init__(self, nnConfig):

		activationFunctions = nnaf.generateActivationFunctions(nnConfig["af_types"])
		sizeOfLayers = nnConfig["layers"]

		nn.Network.__init__(self, sizeOfLayers, activationFunctions)
		self.attRate = nnConfig["attRate"]
		self.repRate = nnConfig["repRate"]
		self.af_types = nnConfig["af_types"]

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

		# logging.debug(vector_r)
		# logging.debug(vector_c)

		max = float("-inf")
		min = float("inf")

		maxTuple = (0, 0, max)
		minTuple = (0, 0, min)

		distanceMatrix = np.zeros((rowNum, colNum), dtype=np.float)
		for r in range(0, rowNum):
			for c in range(0, colNum):
				difference = vector_r[r] - vector_c[c]
				# distance = np.sum(np.square(difference))

				# hamming distance
				distance = np.sum(np.absolute(difference))
				distanceMatrix[r, c] = np.absolute(distance)

				if(distance > max):
					max = distance
					maxTuple = (r, c, max)
				else :
					if (distance < min):
						min = distance
						minTuple = (r, c, min)

		logging.debug("\nThe distance matrix is\n")
		logging.debug(distanceMatrix)

		return maxTuple, minTuple


	def toDict(self):

		jsonObj = dict()
		jsonObj["weightArr"] = []
		jsonObj["biasArr"] = []
		jsonObj["layer_info"] = str(self.sizeOfLayers)
		jsonObj["af_types"] = str(self.af_types)

		for layer in self.layers:
			weight_str = str((layer.weight).tolist())
			bias_str = str((layer.bias).tolist())

			jsonObj["weightArr"].append(weight_str)
			jsonObj["biasArr"].append(bias_str)

		jsonObj["attRate"] = self.attRate
		jsonObj["repRate"] = self.repRate

		return jsonObj



	# two class
	def train(self, dataSet, trainLayerIndex):

		# logging.debug("================================================================================================")
		patterns = dict()
		# trainLayerIndex = 1

		curlayer = self.layers[trainLayerIndex]

		preLayer = self.layers[trainLayerIndex - 1]
		cur_inputs_T = np.transpose(preLayer.outputs)

		returnObj = dict()


		logging.debug("current weight is:")
		logging.debug(curlayer.weight)
		logging.debug("current bias is:")
		logging.debug(curlayer.bias, "\n")


		patterns_input_dict = dict()


		logging.debug("IN THE forward SECTION:============================================================")
		for pair in dataSet:
			key = pair["category"]
			if key not in patterns:
				patterns[key] = []

			output = self.forward(np.transpose(pair["input"]), trainLayerIndex)

			logging.debug("INPUT: %s", str(pair["input"]))
			logging.debug("OUTPUT: %s \n", str(np.transpose(output)))



			patterns_input_dict[output.tostring()] = np.transpose(preLayer.outputs)
			# logging.debug(output)
			patterns[key].append(output)



		logging.debug("IN THE training SECTION====================================================================")
		logging.debug("------------------------------------------")

		# combination of catogries!!
		for keys in list(itertools.combinations_with_replacement(list(patterns.keys()), 2)):
			# logging.debug(keys, patterns[keys[0]], patterns[keys[1]])

			logging.debug("for key: %d, %d", keys[0], keys[1])

			distanceTuple = self.createDistanceMatrix(patterns[keys[0]], patterns[keys[1]])


			the_same_category = (keys[0] == keys[1])
			# logging.debug(keys[0], keys[1])

			# same class
			if the_same_category:
				## same cat
				## min the max distance
				targetIndices = distanceTuple[0]
				multiplier = self.attRate
				logging.debug("max distance: %f" % (targetIndices[2]))

			else:
				# otherwise
				targetIndices = distanceTuple[1]
				multiplier = -1 * self.repRate
				logging.debug("min distance:" %(targetIndices[2]))

			logging.debug("\nThe critical vectors are: \n")


			# the two bad-asses
			vector_p = patterns[keys[0]][targetIndices[0]]
			vector_q = patterns[keys[1]][targetIndices[1]]

			logging.debug(str(np.transpose(vector_p)))
			logging.debug("\nand\n")
			logging.debug(str(np.transpose(vector_q)))

			returnObj[keys] = targetIndices[2]

			# if the_same_category:
			# 	logging.debug("same class:")
			# 	logging.debug(tmp_dict[vector_p.tostring()])
			# 	logging.debug( tmp_dict[vector_q.tostring()] )
			# else:
			# 	logging.debug("not same class:")
			# 	logging.debug(tmp_dict[vector_p.tostring()])
			# 	logging.debug(tmp_dict[vector_q.tostring()] )


			# input("enter.....")




			######################### assume in Y form for now - begin #########################
			diag_p = np.copy(vector_p)
			nn.MatrixOP.apply(diag_p, curlayer.nnFun.derivative)

			diag_q = np.copy(vector_q)
			nn.MatrixOP.apply(diag_q, curlayer.nnFun.derivative)


			######################### assume in Y form for now - end #########################




			## this is the matrix whose diag are the derivatives of neurons

			if not diag_p.shape == (1, 1):
				diag_q = np.diag(np.squeeze(np.asarray(diag_q)))
				diag_p = np.diag(np.squeeze(np.asarray(diag_p)))

			logging.debug("\nTheir diag-directive Matrices are: \n")
			logging.debug(str(diag_p))
			logging.debug("\nand\n")			
			logging.debug(str(diag_q))

			# sensitivity = np.dot(diag_p - diag_q, vector_p - vector_q)

			logging.debug("\nTheir difference Matrices are: \n")
			logging.debug(str(vector_p - vector_q))

			sensitivity_p = np.dot(diag_p, vector_p - vector_q)
			sensitivity_q = np.dot(diag_q, vector_p - vector_q)

			logging.debug("\nTheir sensitivity Matrices are: \n")
			logging.debug(str(sensitivity_p))
			logging.debug("\nand\n")
			logging.debug(str(sensitivity_q))

			logging.debug("\nTheir sensitivity In-flow are: \n")
			logging.debug(str(patterns_input_dict[vector_p.tostring()]))
			logging.debug("\nand\n")
			logging.debug(str(patterns_input_dict[vector_q.tostring()]))


			# logging.debug(sensitivity_p, "\n", patterns_input_dict[vector_p.tostring()])
			# logging.debug(sensitivity_q, "\n", patterns_input_dict[vector_q.tostring()])

			adjust_p = np.dot(sensitivity_p, patterns_input_dict[vector_p.tostring()])
			adjust_q = np.dot(sensitivity_q, patterns_input_dict[vector_q.tostring()])

			logging.debug("\nTheir jusr Matrices are: \n")
			logging.debug(str(adjust_p))
			logging.debug("\nand\n")
			logging.debug(str(adjust_q))

			updateWeight = -1 * multiplier * (adjust_p - adjust_q)

			updateBias = -1 * multiplier * (sensitivity_p - sensitivity_q)

			logging.debug("\nSO The weight updated value is:\n")
			logging.debug(str(updateWeight))
			logging.debug("\nSO The bias updated value is:\n")
			logging.debug(str(updateBias))


		# 	logging.debug(updateWeight)
		# 	logging.debug(updateBias)

		# 	logging.debug("+++++++++++++++++++++++++")

			curlayer.learnWeight(updateWeight)
			curlayer.learnBias(updateBias)

		logging.debug("after this itr, the weight is now:")
		logging.debug(str(curlayer.weight))
		logging.debug("the bias is now:")
		logging.debug(str(curlayer.bias))
		logging.debug("================================================================================================")

		return returnObj




def init(nnConfig):
	return SelfOrganizingMapNeuralNetwork(nnConfig)