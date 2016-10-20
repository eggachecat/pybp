# import numpy as np
# import pylab as plt

# # evenly sampled time at 200ms intervals
# t = np.arange(0., 5., 0.2)
# print(t)
# # red dashes, blue squares and green triangles
# # plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')

# class_O_X = []
# class_O_Y = []

# class_N_X = []
# class_N_Y = []


# data = np.loadtxt("hw1data.dat")
# for row in data:
# 	if row[2] > 0:
# 		class_O_X.append(row[0])
# 		class_O_Y.append(row[1])
# 	else:
# 		class_N_X.append(row[0])
# 		class_N_Y.append(row[1])

# plt.plot(class_O_X, class_O_Y, 'rs', class_N_X, class_N_Y, 'bs')

# # plt.plot(t, t, 'r--')

# plt.show()
import numpy as np
import matplotlib.pyplot as plt
import time

def pltsin(ax, colors=['b']):
    x = np.linspace(0,1,100)
    if ax.lines:
        for line in ax.lines:
            line.set_xdata(x)
            y = np.random.random(size=(100,1))
            line.set_ydata(y)
    else:
        for color in colors:
            y = np.random.random(size=(100,1))
            ax.plot(x, y, color)
    fig.canvas.draw()

fig,ax = plt.subplots(1,1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_xlim(0,1)
ax.set_ylim(0,1)
for f in range(5):
    pltsin(ax, ['b', 'r'])
    time.sleep(1)