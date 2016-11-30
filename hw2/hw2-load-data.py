from pynn import nnSQLite
import os
import numpy as np

exp_id = 1


SQLiteDB = "exp_records.db"
SQLiteDB = os.path.join(os.path.dirname(__file__), SQLiteDB)
nnSQLite.iniGeneralSQLite(SQLiteDB)


obj = nnSQLite.loadFromGeneralDB(exp_id)


print(np.array(eval(obj["biasArr"][0])))