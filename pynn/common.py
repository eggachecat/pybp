from . import nn

import math
import pylab as pl
import numpy as np


def FUNCTION_purelin(x):
	return x

def FUNCTION_DIR_purelin(x):
	return 1

sigmoid = nn.ActivationFunction(lambda x: 1 / (1 + np.exp(-1 * x)), True, lambda y: y * (1-y))
purelin = nn.ActivationFunction(FUNCTION_purelin, False, FUNCTION_DIR_purelin)
output = nn.ActivationFunction(lambda x: x, False, lambda x: 1)
input = nn.ActivationFunction(lambda x: x, False, lambda x: 0)

tanh = nn.ActivationFunction(lambda x: (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)), True, lambda y: (1 - y * y))

sigmoidneg = nn.ActivationFunction(lambda x: -1 + 2 / (1 + np.exp(-1 * x)), True, lambda y: 2 * y * (1-y))
def ac_tanh(c):
	return nn.ActivationFunction(lambda x: (np.exp(c*x) - np.exp(-c*x)) / (np.exp(c*x) + np.exp(-c*x)), True, lambda y: c * (1 - y * y))


