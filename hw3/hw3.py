
######################################################################################################## 
from pynn import nnsimulation 
import pynn.reinforecement as pyr
import numpy as np
from pynn import nnSQLite

import os
config = {
	"acceleration_of_gravity": 9.8,
	"mass_of_cart": 1,
	"mass_of_pole": 0.1,
	"update_time_interval": 0.02,
	"half_length_of_pole": 0.5,
	"force": 10
}


degree_thershold = np.pi / 180

FAILURE_STATES_PARTITION = {
	"x": {
		"-1": lambda x: x < -2.4 or x > 2.4
	},
	"theta": {	
		"-1": lambda x: x < -12 * degree_thershold or x > 12 * degree_thershold
	}
}
# orders matters
stateVariables = ["x", "v_x", "theta", "v_theta"]

STATES_PARTITION = {
	"x": {
		"1": lambda x: x < -0.8,
		"2": lambda x: x < 0.8,
		"3": lambda x: x > 0.8
	},
	"v_x": {
		"0": lambda x: x < -0.5,
		"1": lambda x: x < 0.5,
		"2": lambda x: x > 0.5
	},
	"theta": {	
		"0": lambda x: x < -6 * degree_thershold,
		"1": lambda x: x < -1 * degree_thershold,
		"2": lambda x: x < 0,
		"3": lambda x: x < degree_thershold,
		"4": lambda x: x < 6 * degree_thershold,
		"5": lambda x: True
	},
	"v_theta": {
		"0": lambda x: x < -50 * degree_thershold,
		"1": lambda x: x < 50 * degree_thershold,
		"2": lambda x: True
	}
}

actionSet = [1, -1]


SQLiteDB = "exp_records.db"
SQLiteDB = os.path.join(os.path.dirname(__file__), SQLiteDB)
nnSQLite.iniGeneralSQLite(SQLiteDB)


totalTrails = 100
expCat = ["ase", "ace"]



def ase(totalTrails, greek):
	steps = []
	success = 0
	trail = 0

	r = pyr.ReinforecementNeuralNetwork(["x", "v_x", "theta", "v_theta"], STATES_PARTITION, FAILURE_STATES_PARTITION, actionSet, greek)
	cp = nnsimulation.CartPole(config, figure = False, beta=greek["delta"])


	while trail < totalTrails:

		action = r.get_action()
		direction = action
		cp.update(direction)

		cartPoleState = cp.getState()
		r.setState(cartPoleState)

		if r.ifFailed():

			cp.reset()
			cartPoleState = cp.getState()

			steps.append(success)
			success = 0
			trail += 1

		else:
			success += 1

	nnSQLite.saveToGeneralDB(str(r.qval), str(steps), 0, success, totalTrails, "ase", str(greek), "", "")


def ace(totalTrails, greek):
	steps = []
	success = 0
	trail = 0

	r = pyr.ReinforecementNeuralNetwork(["x", "v_x", "theta", "v_theta"], STATES_PARTITION, FAILURE_STATES_PARTITION, actionSet, greek)
	cp = nnsimulation.CartPole(config, figure = False,  beta=greek["delta"])


	while trail < totalTrails:
		action = r.ace_get_action()
		direction = action
		cp.update(direction)

		cartPoleState = cp.getState()
		r.setState(cartPoleState)

		if r.ace_ifFailed():

			cp.reset()
			cartPoleState = cp.getState()

			steps.append(success)
			success = 0

			trail += 1

		else:
			success += 1

	nnSQLite.saveToGeneralDB(str(r.qval), str(steps), 0, success, totalTrails, "ace", str(greek), "", "")




etas = [0.1, 0.5, 0.9]
deltas = [0, 0.001, 0.01]


for eta in etas:
	for delta in deltas:
		greek = {
			"alpha": 0.5,
			"beta": 0.5,
			"gamma": 0.95,
			"eta": eta,
			"delta": delta
		}


		for x in range(1, 10):
			ace(totalTrails, greek)
			ase(totalTrails, greek)
			print(x)




# # print(r.getBox())

# success = 0
# trail = 0

# best = 0

# while True:

# 	# draw cart
# 	# cp.draw()
	

# 	# get direction from reinforecement neural network
# 	action = r.get_action()

# 	cartPoleState = cp.getState()
# 	# print(cartPoleState)
# 	# print("action is ", action)

# 	direction = action

# 	cp.update(direction)

# 	cartPoleState = cp.getState()
# 	r.setState(cartPoleState)
# 	# print(cartPoleState)

# 	if r.ifFailed():

# 		cp.reset()
# 		cartPoleState = cp.getState()
# 		# r.setState(cartPoleState)
# 		trail += 1

# 		print("stat %d trail:%d" % (trail, success))

# 		# if success > best:
# 		# 	best = success

# 		# 	print("best %d at Trail %d" % (success, trail))

# 		nnSQLite.saveToGeneralDB("", str(greek), 0, success, 1, "reinforement", "", "without ace and noise", trail)

# 		success = 0
# 		# print(r.qval)
# 		# input("input>>")
# 	else:
# 		success += 1

# 	# print("")

# # state = {
# # 	"x": -2.9,
# # 	"v_x": 2.3,
# # 	"theta": 0.02,
# # 	"v_theta": 0.6
# # }

# # r.setState(state)

# # print(r.getBox())













# # randomX = 6 * np.random.random_sample((200,)) - 3
# # randomV_X = 2 * np.random.random_sample((200,)) - 1
# # randomT = (1.57 / 2) * (2 * np.random.random_sample((200,)) - 1)
# # randomV_T = 1.57 * (2 * np.random.random_sample((200,)) - 1)

# # for i in range(0, randomX.size):
# # 	config = {
# # 			"x": randomX[i], "v_x": randomV_X[i], "theta": randomT[i], "v_theta": randomV_T[i]
# 		}
# 	print(config, r.getBox(config))