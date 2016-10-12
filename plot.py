import numpy as np
import pylab as plt

# evenly sampled time at 200ms intervals
t = np.arange(0., 5., 0.2)
print(t)
# red dashes, blue squares and green triangles
# plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')

class_O_X = []
class_O_Y = []

class_N_X = []
class_N_Y = []


data = np.loadtxt("hw1data.dat")
for row in data:
	if row[2] > 0:
		class_O_X.append(row[0])
		class_O_Y.append(row[1])
	else:
		class_N_X.append(row[0])
		class_N_Y.append(row[1])

plt.plot(class_O_X, class_O_Y, 'rs', class_N_X, class_N_Y, 'bs')

# plt.plot(t, t, 'r--')

plt.show()