from pynn import somnn
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


dataSet = nnio.mergeFeatureAndClass(PtFilePath, classFilePath)


# PtFile = open(PtFilePath).readlines()
# classFile = open(classFilePath).readlines()
# file = open(totalFilePath, "w")


# for i in range(0, 100):
# 	content = str(classFile[i])
# 	binStr = content.replace('\t', '')
# 	file.write(PtFile[i].replace('\n', '') + " " + str(int(binStr, 2)) + "\n")

# file.close()



expClassification = "hw1-class-2861"
attRate = 0.01
repRate = 0.1

# cycle of data set
EPOCH = 10


afs = [common.input, common.ac_tanh(1), common.ac_tanh(1), common.ac_tanh(1), common.ac_tanh(1), common.ac_tanh(1)]
layers = [2, 5, 5, 5, 5, 5]

NN = somnn.init(attRate, repRate, layers, afs)
# nnplot.iniGraph(NN, 1)
# nnplot.drawObject(dataSet, True)

NN.train(dataSet)

# while True:
# 	pass