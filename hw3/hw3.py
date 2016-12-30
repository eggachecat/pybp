
######################################################################################################## 
from pynn import nnsimulation 
import pynn.reinforecement as pyr
import numpy as np


config = {
	"acceleration_of_gravity": 9.8,
	"mass_of_cart": 1,
	"mass_of_pole": 0.1,
	"update_time_interval": 0.2,
	"half_length_of_pole": 0.5,
	"force": 1
}

cp = nnsimulation.CartPole(config)


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

actionSet = [-1, 1]

greek = {
	"alpha": 0.5,
	"beta": 0.5,
	"gamma": 0.5,
	"theta": 0.01
}

r = pyr.ReinforecementNeuralNetwork(["x", "v_x", "theta", "v_theta"], STATES_PARTITION, FAILURE_STATES_PARTITION, actionSet, greek)

action = r.getAction()


# print(r.getBox())

success = 0

while True:

	# draw cart
	cp.draw()
	success += 1

	# get direction from reinforecement neural network
	action = r.getAction()

	direction = action

	cp.update(direction)

	cartPoleState = cp.getState()
	r.setState(cartPoleState)

	if r.ifFailed():

		cp.reset()
		cartPoleState = cp.getState()
		r.setState(cartPoleState)

		print("Failed at Trail %d" % (success))
		success = 0
		# print(r.qval)
		input("input>>")



# state = {
# 	"x": -2.9,
# 	"v_x": 2.3,
# 	"theta": 0.02,
# 	"v_theta": 0.6
# }

# r.setState(state)

# print(r.getBox())













# randomX = 6 * np.random.random_sample((200,)) - 3
# randomV_X = 2 * np.random.random_sample((200,)) - 1
# randomT = (1.57 / 2) * (2 * np.random.random_sample((200,)) - 1)
# randomV_T = 1.57 * (2 * np.random.random_sample((200,)) - 1)

# for i in range(0, randomX.size):
# 	config = {
# 			"x": randomX[i], "v_x": randomV_X[i], "theta": randomT[i], "v_theta": randomV_T[i]
# 		}
# 	print(config, r.getBox(config))