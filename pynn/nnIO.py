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

