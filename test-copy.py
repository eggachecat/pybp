from pynn import bpnn
from pynn import nn
from pynn import common
from pynn import nnIO
from pynn import nnplot


import math
import pylab as pl
import numpy as np



# data = np.loadtxt('hw1data.dat', skiprows = 1)
# print( np.asmatrix(data)[:, [0, 1]])

# trainingData, testData = nnIO.readTrainingAndTest('demo.dat', 2, 800)

trainingData, testData = nnIO.readTrainingAndTestData('hw1data-copy.dat', 2, 950)

nnplot.drawData(np.loadtxt("hw1data-copy.dat"))


# pl.ion()
# ax = pl.gca()

# pl.axis([-2, 2, -2, 2])

# global_x = np.arange(-2, 2, 0.05)
# y = global_x

# line = dict()

# line[0], = ax.plot(global_x, y, "r--")
# line[1], = ax.plot(global_x, y, "g--")
# line[2], = ax.plot(global_x, y, "b--")
# line[3], = ax.plot(global_x, y, "y--")

# # for i in range(0,4):
# # 	line[i], = ax.plot(global_x, y, "r--")

# class_O_X = []
# class_O_Y = []

# class_N_X = []
# class_N_Y = []

# data = np.loadtxt("hw1data-copy.dat")

# for row in data:
# 	if row[2] > 0:
# 		class_O_X.append(row[0])
# 		class_O_Y.append(row[1])
# 	else:
# 		class_N_X.append(row[0])
# 		class_N_Y.append(row[1])

# pl.plot(class_O_X, class_O_Y, 'gs', class_N_X, class_N_Y, 'bs')


alpha = 0.01
step = 0.05
alphaEnd = 0.5

afs = [common.input, common.ac_tanh(1), common.ac_tanh(1), common.ac_tanh(1)]
layers = [2, 4, 3, 1]
NN = bpnn.init(alpha, layers, afs)
nnplot.iniGraph(NN, 1)



inputs = trainingData["inputs"]
outputs = trainingData["outputs"]

# test_inputs = testData["inputs"]
# test_outputs = testData["outputs"]
data = nnIO.readInput('hw1data-copy.dat', 2)
test_inputs = data["inputs"]
test_outputs = data["outputs"]

def drawLine(k, b, i):
	
	y = k * global_x + b
	# y = 1 / (1 + np.exp(-1 * y))

	line[i].set_ydata(y)
    # ax.relim()
    # ax.autoscale_view(True,True,True)
	pl.draw()


def drawSegment(NN):
	weights = NN.layers[1].weight
	bias = NN.layers[1].bias

	for i in range(0,4):
		drawLine(-1 * weights[i, 0] / weights[i, 1], -1 * bias[i, 0] / weights[i, 1], i)

	pl.pause(0.00001)


	# pl.show()

# pl.show()	
min = 1000
while(alpha < alphaEnd):
	NN = bpnn.init(alpha, layers, afs)

	# NN.layers[1].weight[0, 0] = 0
	# NN.layers[1].weight[0, 1] = -1
	# NN.layers[1].bias[0, 0] = 0.75

	# NN.layers[1].weight[1, 0] = 0
	# NN.layers[1].weight[1, 1] = -1
	# NN.layers[1].bias[1, 0] = 0.25

	# NN.layers[1].weight[2, 0] = -1
	# NN.layers[1].weight[2, 1] = -0.0000001
	# NN.layers[1].bias[2, 0] = 0.25 

	# NN.layers[1].weight[3, 0] = -1
	# NN.layers[1].weight[3, 1] = -0.0000001
	# NN.layers[1].bias[3, 0] = 0.75


	# NN.layers[1].weight = np.matrix([[ 0, -1],
	# 								 [ 0, -1],
	# 								 [-1, 0.00000001],
	# 								 [-1, 0.00000001]])

	# NN.layers[1].bias = np.matrix([  [ 0.75],
	# 								 [ 0.25],
	# 								 [ 0.25],
	# 								 [ 0.75]])

	# NN.layers[1].weight = np.matrix([[ -0.07522209, -11.35688505],
	# 								 [  0.10955984, -13.40171967],
	# 								 [-12.09067566, 0.21526474],
	# 								 [-11.74733314, -0.10764921]])

	# NN.layers[1].bias = np.matrix([  [ 7.21847827],
	# 								 [ 4.80008582],
	# 								 [ 4.33167249],
	# 								 [ 7.51311826]])

	# NN.layers[2].weight = np.matrix([[-1.42379986,  1.5412965,   1.65160798, -1.59563449],
	# 								 [ 0.0171474 , -0.07079262, -0.12274015,  0.091216  ],
	# 								 [-4.66253457,  4.93983879,  4.91948416, -4.76486111]])

	# NN.layers[2].bias = np.matrix([	 [ 1.13076226],
	# 								 [-0.071814  ],
	# 								 [ 3.65341417]])

	# NN.layers[3].weight = np.matrix([[-0.1545108,  -0.03734087,  1.13278345]])

	# NN.layers[3].bias = np.matrix([[-0.01201823]])


	totalError = 0
	# print(len(inputs))

	for k in range(0, 500):
		totalError = 0
		for i in range(0, len(inputs)):
			inputVector = np.transpose(np.mat(inputs[i]))
			# print(inputVector)	
			output = NN.forward(inputVector)
			teacher = np.transpose(np.mat(outputs[i]))
			NN.backPropagation(teacher - output)
			# print(teacher - output)
			# if( i % 100 == 0):
			# 	drawSegment(NN)

			nnplot.drawNeuron(NN, 1)
			
			output = NN.forward(inputVector)
			teacher = np.transpose(np.mat(test_outputs[i]))
			
			flag = output * teacher > 0
			
			if not flag[0, 0]:
				totalError += 1

		print("alpha = %f, #%d loop, totalError = %d in trails %d " % (alpha, k, totalError, len(test_inputs)))

	alpha += step

	# for x in range(1,4):
	# 	print("weight")
	# 	print(NN.layers[x].weight)
	# 	print("bias")
	# 	print(NN.layers[x].bias)
	# 	print("================================")

print("min = %d" %(min))

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