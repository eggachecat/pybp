from pynn import nnSQLite

from pynn import bpnn
from pynn import nn
from pynn import common
from pynn import nnio
from pynn import nnplot


nnSQLite.iniSQLite("exp_records.db")
# nnSQLite.createTable()


afs = [common.input, common.ac_tanh(1), common.ac_tanh(1), common.ac_tanh(1)]

# nnSQLite.saveToDB(NN, str(afs), "hw1-class", "233", alpha, err_rate)
# print(NN.sizeOfLayers)

# nnSQLite.saveToDB(NN)

while True:
	id = int(input("id>>"))
	NN = nnSQLite.loadFromDB(id, afs)

	while True:
		x = float(input("X>>"))
		y = float(input("Y>>"))
		inputVector = [[x], [y]]
		output = NN.forward(inputVector)
		print("output>>", output)
		c= input("change id ?>>")
		c = str.upper(c)
		if c == "Y" or c == "YES":
			break


# print([[2, 3], [1, 2]])