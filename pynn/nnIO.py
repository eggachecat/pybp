import numpy as np


def readInput(filePath, segement):
	data = np.loadtxt(filePath)

	inputSize = len(data[0])

	inputs = np.asmatrix(data)[:, range(0, segement)]
	outputs = np.asmatrix(data)[:, range(segement, inputSize)]

	return {"inputs": inputs, "outputs": outputs}

# def readTraingData():



def readTrainingAndTest(filePath, segement, inputRows):
	totalData = readInput(filePath, segement)

	if inputRows < 1:
		inputRows = inputRows * len(totalData["inputs"])

	trainingData = dict()
	testData = dict()

	trainingData["inputs"] = totalData["inputs"][0:inputRows, ::]
	testData["inputs"] = totalData["inputs"][inputRows:, ::]

	trainingData["outputs"] = totalData["outputs"][0:inputRows, ::]
	testData["outputs"] = totalData["outputs"][inputRows:, ::]

	return trainingData, testData