from pynn import bpnn
from pynn import nn

import math
import pylab as pl
import numpy as np

afs = [nn.ActivationFunction(lambda x: x, False, lambda x: 1), 
	   nn.ActivationFunction(lambda x: 1 / (1 + np.exp(-1 * x)), True, lambda y: y * (1-y)), 
	   nn.ActivationFunction(lambda x: x, False, lambda x: 1)]
	   
NN = bpnn.init(0.1, [1, 2, 1], afs)

NN.layers[1].weight = np.mat([
	[-0.27], 
	[-0.41]
])

NN.layers[1].bias = np.mat([
	[-0.48], 
	[-0.13]
])

NN.layers[2].weight = np.mat([
	[0.09, -0.17]
])

NN.layers[2].bias = np.mat([
	[0.48]
])

output = NN.forward(np.mat([
	[1.]
]))


teacher = np.mat([
	[1.7071]
])

NN.backPropagation(teacher - output)

print("=====================")
print(NN.layers[2].weight)
print("---------------------")

print(NN.layers[2].bias)
print("---------------------")

print(NN.layers[1].weight)
print("---------------------")

print(NN.layers[1].bias)