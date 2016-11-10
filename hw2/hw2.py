from pynn import bpnn
from pynn import nn
from pynn import common
from pynn import nnio
from pynn import nnplot
from pynn import nnSQLite

import math
import pylab as pl
import numpy as np
import os




##### prepare data
PtFilePath = os.path.join(os.path.dirname(__file__), "hw2pt.dat")
classFilePath = os.path.join(os.path.dirname(__file__), "hw2class.dat")
totalFilePath = os.path.join(os.path.dirname(__file__), "total.dat")



PtFile = open(PtFilePath).readlines()
classFile = open(classFilePath).readlines()
file = open(totalFilePath, "w")


for i in range(0, 100):
	content = str(classFile[i])
	binStr = content.replace('\t', '')
	file.write(PtFile[i].replace('\n', '') + " " + str(int(binStr, 2)) + "\n")

file.close()



expClassification = "hw1-class-2861"
alpha = 0.01
alphaStep = 0.05
maxAplha = 1

# cycle of data set
EPOCH = 100

## (1, 2)
# afs = [common.input, common.ac_tanh(1), common.ac_tanh(1), common.ac_tanh(1)]
# layers = [2, 4, 3, 1]

## (3)
afs = [common.input, common.ac_tanh(1), common.ac_tanh(1), common.ac_tanh(1)]
layers = [2, 8, 6, 1]

NN = bpnn.init(alpha, layers, afs)
nnplot.iniGraph(NN, 1)
nnplot.drawData(np.loadtxt(totalFilePath), True)

# while True:
# 	pass