from pynn import somnn
from pynn import nn
from pynn import common
from pynn import nnio
from pynn import nnplot
from pynn import nnSQLite

import math
import pylab as pl
import numpy as np
import os

import copy

import logging, sys
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
##### prepare data
PtFilePath = os.path.join(os.path.dirname(__file__), "hw2pt-2d.dat")
classFilePath = os.path.join(os.path.dirname(__file__), "hw2class-2d.dat")


dataSet = nnio.mergeFeatureAndClass(PtFilePath, classFilePath)


# PtFile = open(PtFilePath).readlines()
# classFile = open(classFilePath).readlines()
# file = open(totalFilePath, "w")


# for i in range(0, 100):
# 	content = str(classFile[i])
# 	binStr = content.replace('\t', '')
# 	file.write(PtFile[i].replace('\n', '') + " " + str(int(binStr, 2)) + "\n")

# file.close()



expClassification = "hw1-class-2861"
attRate = 0.01
repRate = 1

# cycle of data set
EPOCH = 1000


afs = [common.input, common.ac_tanh(1), common.ac_tanh(5), common.ac_tanh(1), common.ac_tanh(1), common.ac_tanh(1), common.ac_tanh(1)]
layers = [2, 2, 5, 5, 5, 5, 1]

NN = somnn.init(attRate, repRate, layers, afs)
# NN.layers[1].weight = np.ones((5,2))


nnplot.iniGraph(NN, 1)
# nnplot.clf()
nnplot.drawObject(dataSet, -1)

# style = ["rs", "gs", "bs", "ks", "ro", "go", "bo", "ko", "r^", "g^", "b^", "k^"]

style = nnplot.__shuffle_colors

pl.figure(2)
pl.ylim([-2,2])
pl.xlim([-2,2])
for i in range(0, len(dataSet)):
	pair = dataSet[i]
	pl.scatter(pair["input"][0, 0], pair["input"][0, 1], c=style[i])

pl.figure(3)
for trainLayerIndex in range(1, 2):
	print("to layer", trainLayerIndex)
	for x in range(1, EPOCH):

		# nnplot.drawNeuron(NN, 1)
		NN.train(dataSet, trainLayerIndex)


		tmpDataSet = copy.deepcopy(dataSet)

		pairInt = dict()
		pairCounter = 0

		nnplot.clf()

		for i in range(0, len(tmpDataSet)):

			pair = tmpDataSet[i]
			result =  np.transpose(NN.forward(np.transpose(pair["input"]), trainLayerIndex))
			# result = (result > 0).astype(int)[0]
			# result = ''.join(map(str, result))
			# result = int(result, 2)

			# if not result in pairInt.keys():
			# 	pairInt[result] = pairCounter
			# 	pairCounter += 1

			# pair["category"] = pairInt[result]

			pl.scatter(result[0, 0], result[0, 1], c=style[i])
			# pl.plot(pair["input"][0, 0], pair["input"][0, 1], style[i])


		# nnplot.drawObject(tmpDataSet, 0.0001)
		input("enter.....")
		nnplot.clf()

		# pl.figure(2)

# counter = 0
# correct = 0
# for pair in tmpDataSet:
# 	counter += 1
# 	result =  np.transpose(NN.forward(np.transpose(pair["input"]), 6))
# 	result = result > 0

# 	# print(pair["category"], result[0,0])
# 	if(bool(pair["category"]) ==  result[0,0]):
# 		correct += 1

# print(counter, correct)

		# print("weight", NN.layers[trainLayerIndex].weight)
		# nnplot.drawObject(dataSet, -1)
		# nnplot.drawObject(tmpDataSet, 0.0001, ["r^", "g^", "b^"])
		# nnplot.clf()
		# nnplot.drawObject(dataSet, False)

input("enter.....")

# while True:
# 	pass