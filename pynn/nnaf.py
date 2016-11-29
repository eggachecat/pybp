from . import nn

import math
import pylab as pl
import numpy as np






def sigmoid(a = 1):
	return nn.ActivationFunction(
		lambda x: 1 / (1 + np.exp(-1 * a * x)), True, 
		lambda y: a * y * (1-y))

def purelin(a):
	return nn.ActivationFunction(lambda x: x, False, lambda x: 1)


# purelin = nn.ActivationFunction(lambda x: x, False, lambda x: 1)
# output = nn.ActivationFunction(lambda x: x, False, lambda x: 1)
# input = nn.ActivationFunction(lambda x: x, False, lambda x: 0)

# tanh = nn.ActivationFunction(lambda x: (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)), True, lambda y: (1 - y * y))

# sigmoidneg = nn.ActivationFunction(lambda x: -1 + 2 / (1 + np.exp(-1 * x)), True, lambda y: 2 * y * (1-y))
def tanh(a = 1):
	return nn.ActivationFunction(
		lambda x: (np.exp(a*x) - np.exp(-a*x)) / (np.exp(a*x) + np.exp(-a*x)), True, 
		lambda y: a * (1 - y * y))


def generateActivationFunctions(afsArr):

	afs = []
	_afset = globals()
	fun = None

	for item in afsArr:
		if type(item) is tuple:
			funName = item[0]
			funParm = item[1]
			fun = _afset[funName](funParm)
			afs.append(fun)
		else:
			if (not type(item) is int) or (item < 0) or (not fun):
				print("something wrong with the activation-function-config")
				exit()

			for x in range(0, item):
				afs.append(fun)

	return afs
