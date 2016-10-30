from pynn import bpnn
from pynn import nn
from pynn import common
from pynn import nnIO
from pynn import nnplot


import math
import pylab as pl
import numpy as np


dataFileName = 'hw1data.dat'

alpha = 0.01
alphaStep = 0.05
maxAplha = 0.5

# cycle of data set
EPOCH = 100


trainingData, testData = nnIO.readTrainingAndTestData(dataFileName, 2, 950)

inputs = trainingData["inputs"]
outputs = trainingData["outputs"]

data = nnIO.readInput(dataFileName, 2)
test_inputs = data["inputs"]
test_outputs = data["outputs"]

afs = [common.input, common.ac_tanh(1), common.ac_tanh(1), common.ac_tanh(1)]
layers = [2, 4, 3, 1]

NN = bpnn.init(alpha, layers, afs)
nnplot.iniGraph(NN, 1)
nnplot.drawData(np.loadtxt(dataFileName))


while(alpha < maxAplha):

	NN = bpnn.init(alpha, layers, afs)
	totalError = 0
	
	for k in range(0, EPOCH):
		totalError = 0
		for i in range(0, len(inputs)):
			inputVector = np.transpose(np.mat(inputs[i]))
			output = NN.forward(inputVector)
			teacher = np.transpose(np.mat(outputs[i]))
			errorVector = teacher - output
			NN.backPropagation(errorVector)
		# update every cycle of trainig data
		nnplot.drawNeuron(NN, 1)
			

		for i in range(0, len(test_inputs)):

			inputVector = np.transpose(np.mat(test_inputs[i]))
			output = NN.forward(inputVector)
			teacher = np.transpose(np.mat(test_outputs[i]))

			flag = output * teacher > 0

			if not flag[0, 0]:
				totalError += 1

		print("alpha = %f, #%d loop, totalError = %d in trails %d " % (alpha, k, totalError, len(test_inputs)))

	alpha += alphaStep

