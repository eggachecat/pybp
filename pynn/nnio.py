import numpy as np
import random

def readInput(filePath, segement, random = False):
	data = np.loadtxt(filePath)

	inputSize = len(data[0])

	if random:
		random.shuffle(data)

	inputs = np.asmatrix(data)[:, range(0, segement)]
	outputs = np.asmatrix(data)[:, range(segement, inputSize)]

	return {"inputs": inputs, "outputs": outputs}

def mergeFeatureAndClass(featurePath, classPath, inputRows = 0, random = False):
	featureData = np.loadtxt(featurePath)
	classData = open(classPath).readlines()

	if len(featureData) != len(classData):
		print("feature and class data are illeagel")
		exit(0)

	catCounter = 0
	catList = dict()

	mergedData = []

	for i in range(0,len(featureData)):

		if classData[i] not in catList:
			catList[classData[i]] = catCounter
			catCounter += 1

		mergedData.append({
			"input": np.asmatrix(featureData[i]),
			"category": catList[classData[i]]
		})

	return mergedData

		



def readTrainingAndTestData(filePath, segement, inputRows, random = False):
	totalData = readInput(filePath, segement)

	if inputRows < 1:
		inputRows = inputRows * len(totalData["inputs"])

	trainingData = dict()
	testData = dict()

	allInputs = totalData["inputs"]
	allOutputs = totalData["outputs"]

	trainingData["inputs"] = allInputs[0:inputRows, ::]
	testData["inputs"] = allInputs[inputRows: , ::]

	trainingData["outputs"] = allOutputs[0:inputRows, ::]
	testData["outputs"] = allOutputs[inputRows:, ::]

	return trainingData, testData

