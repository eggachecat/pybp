from pynn import bpnn
from pynn import nn
from pynn import common
from pynn import nnio
from pynn import nnplot
from pynn import nnSQLite

import math
import pylab as pl
import numpy as np
import os

##### init data
dataFileName = 'hw1data.dat'
dataFileName = os.path.join(os.path.dirname(__file__), dataFileName)
SQLiteDB = "exp_records.db"
SQLiteDB = os.path.join(os.path.dirname(__file__), SQLiteDB)
nnSQLite.iniSQLite(SQLiteDB)


expClassification = "hw1-class-2861"
alpha = 0.01
alphaStep = 0.05
maxAplha = 0.1

# cycle of data set
EPOCH = 10


af_types = [("purelin", 1), ("tanh", 1), 10]


###########################

trainingData, testData = nnio.readTrainingAndTestData(dataFileName, 2, 950)

inputs = trainingData["inputs"]
outputs = trainingData["outputs"]

data = nnio.readInput(dataFileName, 2)
test_inputs = data["inputs"]
test_outputs = data["outputs"]

expCtr = 0;


def calculateError(NN, test_inputs, test_outputs):
	totalError = 0
	for i in range(0, len(test_inputs)):

		inputVector = np.transpose(np.mat(test_inputs[i]))
		output = NN.forward(inputVector)
		teacher = np.transpose(np.mat(test_outputs[i]))

		flag = output * teacher > 0

		if not flag[0, 0]:
			totalError += 1
	return totalError


while(alpha < maxAplha):



	nnConfig = {
		"alpha": alpha,
		"layers": layers,
		"af_types": af_types
	}

	NN = bpnn.init(nnConfig)
	nnplot.iniNeurons(NN)

	totalError = 0
	recordError = []
	
	for k in range(0, EPOCH):
		for i in range(0, len(inputs)):
			inputVector = np.transpose(np.mat(inputs[i]))
			output = NN.forward(inputVector)
			teacher = np.transpose(np.mat(outputs[i]))
			errorVector = teacher - output
			NN.backPropagation(errorVector)
		# update every cycle of trainig data
		nnplot.updateNeuron(NN, 1)
			
  
		totalError = calculateError(NN, test_inputs, test_outputs)

		errorRate = float(totalError/len(test_inputs))
		recordError.append(errorRate)
		exp_note = "alpha = %f, #%d loop, total error = %d in trails = %d, errorRate = %f" % (alpha, k, totalError, len(test_inputs), errorRate)
		exp_id = nnSQLite.saveToDB(NN, "[common.input, common.ac_tanh(1), common.ac_tanh(1), common.ac_tanh(1)]", expClassification, exp_note, alpha, errorRate)
		print(exp_note + " at exp_id = " + str(exp_id))

	pl.figure(2)
	pl.plot(range(1, EPOCH + 1), recordError, color = expCtr)
	pl.title('EPOCH - ErrorRate')
	pl.xlabel('EPOCH')
	pl.ylabel('error rate')

	alpha += alphaStep
	expCtr += 1

nnSQLite.closeDB()
