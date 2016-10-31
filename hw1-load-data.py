import numpy as np
from pynn import nnSQLite
from pynn import bpnn
from pynn import nn
from pynn import common
from pynn import nnio
from pynn import nnplot


nnSQLite.iniSQLite("exp_records.db")
# nnSQLite.createTable()


dataFileName = 'hw1data.dat'

afs = [common.input, common.ac_tanh(1), common.ac_tanh(1), common.ac_tanh(1)]

# nnSQLite.saveToDB(NN, str(afs), "hw1-class", "233", alpha, err_rate)
# print(NN.sizeOfLayers)

# nnSQLite.saveToDB(NN)


def drawLines():

	id = int(input("id>>"))
	NN = nnSQLite.loadFromDB(id, afs)
	nnplot.iniGraph(NN, 1)
	nnplot.drawData(np.loadtxt(dataFileName))
	nnplot.drawNeuron(NN, 1)
	input("enter anything to quit")

def testData():
	id = int(input("id>>"))
	NN = nnSQLite.loadFromDB(id, afs)

	while True:
		x = float(input("X>>"))
		y = float(input("Y>>"))
		inputVector = [[x], [y]]
		output = NN.forward(inputVector)
		print("output>>", output)

def showOutputs():
	id = int(input("id>>"))
	NN = nnSQLite.loadFromDB(id, afs)

	while True:
		x = float(input("X>>"))
		y = float(input("Y>>"))
		inputVector = [[x], [y]]
		output = NN.forward(inputVector)
		for i in range(1,NN.nnDepth):
			print("#layer-%d" %(i))
			print(NN.layers[i].outputs)
			print("\n")

# showOutputs()
drawLines()
# testData()