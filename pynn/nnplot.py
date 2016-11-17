import pylab as pl
import numpy as np

# only for 2-D data !
# assume (x,y, class)

global __global_X 
global __global_lines


def drawObject(objList, pause, style = ["rs", "gs", "bs", "r^", "g^", "b^"]):

	for obj in objList:
		cor = obj["input"]
		try:
			pl.plot(cor[0, 0], cor[0, 1], style[obj["category"]])
		except IndexError:
			print("The length of color-style-array smaller than the number of categories!! Please specify the color-style-array!!")
			exit()

	if pause:
		pl.pause(0)


def drawData(data, pause = False, classIndex = 2, style = ["rs", "gs", "bs", "r^", "g^", "b^"]):

	categoryStyleDict = dict()
	styleCtr = 0

	for row in data:
		key = row[classIndex]
		if not categoryStyleDict.get(key):
			try:
				categoryStyleDict[key] = style[styleCtr]
				styleCtr += 1
			except IndexError:
				print("The length of color-style-array smaller than the number of categories!! Please specify the color-style-array!!")
				exit()

		catStyle = categoryStyleDict[key]
		pl.plot(row[0], row[1], catStyle)
	
	if pause:
		pl.pause(0)

def drawLine(k, b, line):

	global __global_X
	y = k * __global_X + b
	line.set_ydata(y)
	pl.draw()


def drawNeuron(NN, layerNumber):


	global __global_lines
	target = NN.layers[layerNumber]
	weight = target.weight
	bias = target.bias

	lineNumbers = weight.shape[0]

	for i in range(0, lineNumbers):
		k = -1 * weight[i, 0] / weight[i, 1]
		b = -1 * bias[i, 0] / weight[i, 1]
		line = __global_lines[i]
		drawLine(k, b, line)

	pl.pause(0.00001)

def iniGraph(NN, layerNumber, axis = [-2, 2, -2, 2], step = 0.05, lineStyle=["r--", "g--", "b--", "y--", "m--", "c--", "k--", "r--"]):
	
	global __global_X
	global __global_lines
	
	__global_X = []
	__global_lines = dict()
	# init graph
	pl.ion()
	ax = pl.gca()
	pl.axis(axis)

	target = NN.layers[layerNumber]
	weight = target.weight

	numberOfLines = weight.shape[0]

	__global_X = np.arange(axis[0], axis[1], step)

	# init lines
	for i in range(0, numberOfLines):
		__global_lines[i], = ax.plot(__global_X, __global_X, lineStyle[i])

