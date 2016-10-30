import sqlite3
from pynn import common
from pynn import bpnn


global __conn
global __cursor
def createTable():

	global __conn, __cursor

	try:
		__cursor.execute('''CREATE TABLE exp_info
		             (exp_id INTEGER PRIMARY KEY NOT NULL, alpha REAL, err_rate REAL, layer_info TEXT, act_fun TEXT, exp_category TEXT, exp_note TEXT)''')


		__cursor.execute('''CREATE TABLE exp_weight
		             (exp_weight_id INTEGER PRIMARY KEY NOT NULL, exp_id INTEGER, weight_seq INTEGER, weight_str TEXT)''')

		__cursor.execute('''CREATE TABLE exp_bias
		             (exp_bias_id INTEGER PRIMARY KEY NOT NULL, exp_id INTEGER, bias_seq INTEGER, bias_str TEXT)''')

		__conn.commit()
	except sqlite3.DatabaseError:
		print("Tables exists.")
		pass

def iniSQLite(dbPath):

	global __conn, __cursor

	__conn = sqlite3.connect(dbPath)
	__conn.row_factory = sqlite3.Row
	__cursor = __conn.cursor()


def saveToDB(NN, act_fun, exp_category, exp_note, alpha, err_rate):

	global __conn, __cursor

	# print("save db")
	# for layer in NN.layers:
	# 	print(layer.weight)

	layer_info = str(NN.sizeOfLayers)
	
	__cursor.execute('''INSERT INTO exp_info(alpha, err_rate, layer_info, act_fun, exp_category, exp_note)
				 VALUES(?, ?, ?, ?, ?, ?)''', (alpha, err_rate, layer_info, act_fun, exp_category, exp_note))

	exp_id = __cursor.lastrowid

	index = 0
	for layer in NN.layers:

		weight_str = str((layer.weight).tolist())

		bias_str = str((layer.bias).tolist())

		__cursor.execute('''INSERT INTO exp_weight(exp_id, weight_seq, weight_str)
					 VALUES(?, ?, ?)''', (exp_id, index, weight_str))

		__cursor.execute('''INSERT INTO exp_bias(exp_id, bias_seq, bias_str)
					 VALUES(?, ?, ?)''', (exp_id, index, bias_str))

		index += 1

	__conn.commit()

	
	return exp_id

def closeDB():
	__conn.close()

def loadFromDB(id, afs):
	__cursor.execute('''SELECT * FROM exp_info WHERE exp_id = :id''', [id])

	exp_info = __cursor.fetchone()
	layers = eval(exp_info["layer_info"])
	alpha = exp_info["alpha"]

	NN = bpnn.init(alpha, layers, afs)

	__cursor.execute('''SELECT * FROM exp_weight WHERE exp_id = :id ORDER BY weight_seq''', [id])
	exp_weight = __cursor.fetchall()

	__cursor.execute('''SELECT * FROM exp_bias WHERE exp_id = :id ORDER BY bias_seq''', [id])
	exp_bias = __cursor.fetchall()

	index = 0
	for layer in NN.layers:
		layer.weight = eval(exp_weight[index]["weight_str"])
		layer.bias = eval(exp_bias[index]["bias_str"])
		index += 1

	return NN
	# print(exp_weight["weight_str"])
	# weight = eval(exp_weight["weight_str"])

