from pynn import bpnn
from pynn import nn
from pynn import common
from pynn import nnio
from pynn import nnplot
from pynn import nnSQLite

import math
import pylab as pl
import numpy as np

##### init data
dataFileName = 'hw1data.dat'
expClassification = "hw1-class-2861"
alpha = 0.01
alphaStep = 0.05
maxAplha = 1

# cycle of data set
EPOCH = 100

## (1, 2)
# afs = [common.input, common.ac_tanh(1), common.ac_tanh(1), common.ac_tanh(1)]
# layers = [2, 4, 3, 1]

## (3)
afs = [common.input, common.ac_tanh(1), common.ac_tanh(1), common.ac_tanh(1)]
layers = [2, 8, 6, 1]

NN = bpnn.init(alpha, layers, afs)
nnplot.iniGraph(NN, 1)
nnplot.drawData(np.loadtxt(dataFileName))

nnSQLite.iniSQLite("exp_records.db")
###########################

trainingData, testData = nnio.readTrainingAndTestData(dataFileName, 2, 950)

inputs = trainingData["inputs"]
outputs = trainingData["outputs"]

data = nnio.readInput(dataFileName, 2)
test_inputs = data["inputs"]
test_outputs = data["outputs"]


while(alpha < maxAplha):

	NN = bpnn.init(alpha, layers, afs)
	totalError = 0
	recordError = []
	
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

		errorRate = float(totalError/len(test_inputs))
		recordError.append(errorRate)
		exp_note = "alpha = %f, #%d loop, total error = %d in trails = %d, errorRate = %f" % (alpha, k, totalError, len(test_inputs), errorRate)
		exp_id = nnSQLite.saveToDB(NN, "[common.input, common.ac_tanh(1), common.ac_tanh(1), common.ac_tanh(1)]", expClassification, exp_note, alpha, errorRate)
		print(exp_note + " at exp_id = " + str(exp_id))

	pl.figure(2)
	pl.plot(range(1, EPOCH+1), recordError)
	pl.title('EPOCH - ErrorRate')
	pl.xlabel('EPOCH')
	pl.ylabel('error rate')

	alpha += alphaStep

nnSQLite.closeDB()
nnSQLite.iniSQLite("exp_records.db")

while True:
	id = int(input("id>>"))
	NN = nnSQLite.loadFromDB(id, afs)

	while True:
		x = float(input("X>>"))
		y = float(input("Y>>"))
		inputVector = [[x], [y]]
		output = NN.forward(inputVector)
		print("output>>", output)
		c= input("change id ?>>")
		c = str.upper(c)
		if c == "Y" or c == "YES":
			break
nnSQLite.closeDB()