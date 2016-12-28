from pynn import nn
from pynn import nnaf

import math
import pylab as pl
import numpy as np


import itertools

import logging


class HopfieldLayer(nn.Layer):

	pass


class HopfielNeuralNetwork(nn.Network):

	type = "hopfield"

	def __init__(self, nnConfig):
		pass

	
	def forward(self, inputVector, maxDepth):

	

	def createDistanceMatrix(self, vector_r, vector_c):



	def createRandomDistanceMatrix(self, vector_r, vector_c):



	def toDict(self):


	# two class
	def train(self, dataSet, trainLayerIndex, randomSelection = False):

		# logging.debug("================================================================================================")




def init(nnConfig):
	return HopfielNeuralNetwork(nnConfig)