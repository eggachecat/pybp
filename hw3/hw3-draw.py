import pylab as pl

import numpy as np


from pynn import nnSQLite

import os

import json

SQLiteDB = "exp_records.db"
SQLiteDB = os.path.join(os.path.dirname(__file__), SQLiteDB)
nnSQLite.iniGeneralSQLite(SQLiteDB)







def draw(ace, ase, fileName):

	x = range(0, len(ace))
	print(len(ace))
	pl.plot(x, ace, label='ase', color = "red")
	pl.plot(x, ase, label='ace', color = "blue")


	pl.title('learning cureves')
	pl.xlabel('trials')
	pl.ylabel('time step util failure')
	pl.legend(loc='upper center', fancybox=True, shadow=True, ncol = 2)


	path = os.path.join(os.path.dirname(__file__), "outputs/%d.png" % (fileName))
	pl.savefig(path)
	pl.close()

minIt = 127
maxIt = 133

import re
for x in range(minIt, maxIt, 2):
	
	column = "initial_parameters_json"
	aceStr = nnSQLite.selectFromGeneralDB("SELECT %s FROM exp_info WHERE exp_id=%d AND epochs = 100" %(column, x))
	aseStr = nnSQLite.selectFromGeneralDB("SELECT %s FROM exp_info WHERE exp_id=%d AND epochs = 100" %(column, x+1))

	ace = eval(aceStr)
	ase = eval(aseStr)
	# ace =  eval(aceStr[0])
	# print(type(ace))
	# ase =  np.array(eval(aseStr[0]))

	# print(ace[0])
	draw(ace, ase, x)

