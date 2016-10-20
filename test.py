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



class_O_X = []
class_O_Y = []

class_N_X = []
class_N_Y = []

data = np.loadtxt("hw1data.dat")

for row in data:
	if row[2] > 0:
		class_O_X.append(row[0])
		class_O_Y.append(row[1])
	else:
		class_N_X.append(row[0])
		class_N_Y.append(row[1])

pl.plot(class_O_X, class_O_Y, 'gs', class_N_X, class_N_Y, 'bs')

alpha = 0.1
step = 0.05

afs = [common.purelin, common.sigmoid, common.sigmoid, common.sigmoid]
layers = [2, 4, 3, 1]
NN = bpnn.init(alpha, layers, afs)



inputs = trainingData["inputs"]
outputs = trainingData["outputs"]

test_inputs = testData["inputs"]
test_outputs = testData["outputs"]


def drawLine(k, b):
	pl.plot(class_O_X, class_O_Y, 'gs', class_N_X, class_N_Y, 'bs')

	x = np.arange(0, 1, 0.1)
	print(k, b)
	y = k*x + b

	y = 1 / (1 + np.exp(-1 * y))

	pl.plot(x, y, 'rs')

def drawSegment(NN):
	weights = NN.layers[1].weight
	bias = NN.layers[1].bias

	for i in range(0,4):
		drawLine(-1 * weights[i, 0] / weights[i, 1], -1 * bias[i, 0] / weights[i, 1])

	pl.axis([0, 1, 0, 1])

	pl.show()

# pl.show()	

while(alpha < 0.5) :
	NN = bpnn.init(alpha, layers, afs)

	# weights = NN.layers[1].weight
	# bias = NN.layers[1].bias

	# for i in range(0,4):
	# 	drawLine(-1 * weights[i, 0] / weights[i, 1], -1 * bias[i, 0] / weights[i, 1])

	# pl.axis([-2, 2, -2, 2])

	# pl.show()
		
	# drawSegment(NN)


	totalError = 0

	for i in range(0, len(inputs)):
		inputVector = np.transpose(np.mat(inputs[i]))
		output = NN.forward(inputVector)

		teacher = np.transpose(np.mat(outputs[i]))
		NN.backPropagation(teacher - output)
	drawSegment(NN)
	for i in range(0, len(inputs)):
		inputVector = np.transpose(np.mat(inputs[i]))
		output = NN.forward(inputVector)

		teacher = np.transpose(np.mat(outputs[i]))
		NN.backPropagation(teacher - output)
	drawSegment(NN)
	for i in range(0, len(inputs)):
		inputVector = np.transpose(np.mat(inputs[i]))
		output = NN.forward(inputVector)

		teacher = np.transpose(np.mat(outputs[i]))
		NN.backPropagation(teacher - output)
	drawSegment(NN)
	for i in range(0, len(inputs)):
		inputVector = np.transpose(np.mat(inputs[i]))
		output = NN.forward(inputVector)

		teacher = np.transpose(np.mat(outputs[i]))
		NN.backPropagation(teacher - output)	
	drawSegment(NN)


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