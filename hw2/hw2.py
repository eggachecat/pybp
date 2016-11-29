from pynn import somnn
from pynn import nn
from pynn import common
from pynn import nnaf
from pynn import nnio
from pynn import nnplot
from pynn import nnSQLite

import math
import pylab as pl
import numpy as np
import os

import time
import copy

import logging, sys
logging.basicConfig(stream=sys.stderr, level=logging.INFO)


dataset_name = "hw2"

##### prepare data
PtFilePath = os.path.join(os.path.dirname(__file__), "hw2pt.dat")
classFilePath = os.path.join(os.path.dirname(__file__), "hw2class.dat")

SQLiteDB = "exp_all_in_one_records.db"
SQLiteDB = os.path.join(os.path.dirname(__file__), SQLiteDB)
nnSQLite.iniGeneralSQLite(SQLiteDB)




dataSet = nnio.mergeFeatureAndClass(PtFilePath, classFilePath)

# cycle of data set
# epoch = 5000


def test(nnConfig, epoch):


	sizeOfLayers = len(nnConfig["layers"])

	timestr = time.strftime("%Y%m%d-%H%M%S")
	recordDir = os.path.join(os.path.dirname(__file__ ) + "\exp-figures\\")
	expDir = os.path.join(recordDir + timestr)

	if not os.path.exists(expDir):
	    os.makedirs(expDir)
	JsonPath = os.path.join(expDir, "config.json")


	NN = somnn.init(nnConfig)
	ini_nn_dict = NN.toDict() 

	outputIndex = sizeOfLayers - 1
	initialErrorRate = calculateErrorRate(NN, dataSet, outputIndex)



	nnplot.iniGraph(NN, 1, ion = False)
	nnplot.drawObject(dataSet, -1)


	for trainLayerIndex in range(1, sizeOfLayers):
		# print("to layer", trainLayerIndex)

		diffCurve = dict()

		for x in range(0, epoch):
			diffObj = NN.train(dataSet, trainLayerIndex)

			for key, value in diffObj.items():
				if not key in diffCurve:
					diffCurve[key] = []

				diffCurve[key].append(value)


			if x % 10 == 0:
				tmpDataSet = copy.deepcopy(dataSet)

				pairInt = dict()
				pairCounter = 0

				for pair in tmpDataSet:
					result =  np.transpose(NN.forward(np.transpose(pair["input"]), trainLayerIndex))
					result = (result > 0).astype(int)[0]
					result = ''.join(map(str, result))
					result = int(result, 2)

					if not result in pairInt.keys():
						pairInt[result] = pairCounter
						pairCounter += 1

					pair["category"] = pairInt[result]

				pl.figure(1 + trainLayerIndex)
				# nnplot.clf()
				nnplot.drawObject(tmpDataSet, -1)
				# input("enter.....")
				# nnplot.clf()

		ctr = 0
		for key, yArr in diffCurve.items():
			pl.figure(ctr + 2 * sizeOfLayers, figsize=(18.0, 12.0))

			# color index added to due to hope of easy-distinguished
			pl.plot(range(1, epoch + 1), yArr, label='$Layer {i}$'.format(i = trainLayerIndex), color = nnplot.__shuffle_colors[trainLayerIndex])
			pl.title('epoch - Difference between ' + str(key))
			pl.ylabel('Difference')
			pl.legend(loc='upper center', bbox_to_anchor=(0.5, -0.02), fancybox=True, shadow=True, ncol = 10)

			ctr += 1

	# nnio.saveResultToJson(JsonPath, NN.toJson())




	for x in range(2, sizeOfLayers + 1):
		pl.figure(x)
		figPath = os.path.join(expDir, "layer-%d.png"%(x-1))
		pl.savefig(figPath)
		pl.close()

	ctr = 0
	for key, yArr in diffCurve.items():
		pl.figure(ctr + 2 * sizeOfLayers)
		figPath = os.path.join(expDir, "%s.png"%(str(key)))
		pl.savefig(figPath)
		pl.close()
		ctr += 1

	errorRate = calculateErrorRate(NN, dataSet, outputIndex)
	print(initialErrorRate, errorRate)
	exp_note = timestr
	exp_id = nnSQLite.saveToGeneralDB(NN.toDict(), ini_nn_dict, initialErrorRate, errorRate, epoch, "classification", dataset_name, exp_note, timestr)


def calculateErrorRate(NN, dataSet, outputLayer):
	errCtr = 0
	ctr = 0
	for pair in dataSet:
		ctr += 1
		key = pair["category"]
		output = NN.forward(np.transpose(pair["input"]), outputLayer)
		result = (output > 0).astype(int)[0]
		result = ''.join(map(str, result))
		result = int(result, 2)
		if not result == key:
			errCtr += 1

	errorRate = errCtr / ctr
	errorRate = min(errorRate, 1 - errorRate)

	return errorRate




af_types = [("purelin", 1), ("tanh", 1), 15]



epochs = [2]
times = 1
layersBox = [[2, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]]
step = (0.0001, 1)


for epoch in epochs:
	for layers in layersBox:
		for x in range(0, times):
			nnConfig = {
				"attRate": step[0],
				"repRate": step[1],
				"af_types": af_types,
				"layers": layers
			}
			test(nnConfig, epoch)



# for layers in layersBox:
# 	for step in stepBox:

# 		nnConfig = {
# 			"attRate": step[0],
# 			"repRate": step[1],
# 			"af_types": af_types,
# 			"layers": layers
# 		}
# 		print(nnConfig)
# 		test(nnConfig)

nnSQLite.closeDB()