# """
# A simple example of an animated plot
# """
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation

# fig, ax = plt.subplots()

# x = np.arange(0, 2*np.pi, 0.01)        # x-array
# line, = ax.plot(x, np.sin(x))

# def animate(i):
#     line.set_ydata(np.sin(x+i/10.0))  # update the data
#     return line,

# #Init only required for blitting to give a clean slate.
# def init():
#     line.set_ydata(np.ma.array(x, mask=True))
#     return line,

# ani = animation.FuncAnimation(fig, animate, np.arange(1, 200), init_func=init,
#     interval=25, blit=False)
# plt.show()

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 10, 0.1)
y = np.random.randn(100)

plt.ion()
ax = plt.gca()
# ax.set_autoscale_on(True)
line, = ax.plot(x, y)


def drawline(k, b):
	y = k * x + b
	line.set_ydata(y)
	# ax.relim()
	# ax.autoscale_view(True,True,True)
	plt.draw()


for i in range(0, 100):
    drawline(2 + 0.1 * i, 1)
    plt.pause(0.01)
