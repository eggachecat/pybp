from pynn import bpnn
from pynn import nn
from pynn import common
from pynn import nnIO

import math
import pylab as pl
import numpy as np



# data = np.loadtxt('hw1data.dat', skiprows = 1)
# print( np.asmatrix(data)[:, [0, 1]])

# trainingData, testData = nnIO.readTrainingAndTest('demo.dat', 2, 800)

trainingData, testData = nnIO.readTrainingAndTest('hw1data.dat', 2, 800)



alpha = 0.1
step = 0.05

afs = [common.purelin, common.sigmoid, common.sigmoid, common.sigmoid]
layers = [2, 4, 3, 1]
NN = bpnn.init(alpha, layers, afs)



inputs = trainingData["inputs"]
outputs = trainingData["outputs"]

test_inputs = testData["inputs"]
test_outputs = testData["outputs"]


while(alpha < 0.99) :
	NN = bpnn.init(alpha, layers, afs)

	totalError = 0

	for i in range(0, len(inputs)):
		inputVector = np.transpose(np.mat(inputs[i]))
		output = NN.forward(inputVector)

		teacher = np.transpose(np.mat(outputs[i]))
		NN.backPropagation(teacher - output)



	for i in range(0, len(test_inputs)):
		inputVector = np.transpose(np.mat(test_inputs[i]))
		output = NN.forward(inputVector)
		
		teacher = np.transpose(np.mat(test_outputs[i]))

		err = (teacher - output)
		
		totalError += err[0,0]

	print("alpha = %f, totalError = %f" % (alpha, totalError))
	alpha += step



# NN.layers[1].weight = np.mat([
# 	[-0.27], 
# 	[-0.41]
# ])

# NN.layers[1].bias = np.mat([
# 	[-0.48], 
# 	[-0.13]
# ])

# NN.layers[2].weight = np.mat([
# 	[0.09, -0.17]
# ])

# NN.layers[2].bias = np.mat([
# 	[0.48]
# ])

# output = NN.forward(np.mat([
# 	[1.]
# ]))


# teacher = np.mat([
# 	[1.7071]
# ])

# NN.backPropagation(teacher - output)

# print("=====================")
# print(NN.layers[2].weight)
# print("---------------------")

# print(NN.layers[2].bias)
# print("---------------------")

# print(NN.layers[1].weight)
# print("---------------------")

# print(NN.layers[1].bias)