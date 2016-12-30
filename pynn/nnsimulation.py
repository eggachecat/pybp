import numpy as np
import pylab as pl
import matplotlib.patches as patches
import matplotlib.animation as animation
import matplotlib.lines as lines                        


class CartPole():

	CART_LEVEL = 0.0
	CART_WIDTH = 2.0
	CART_HEIGHT = 1.0
	POLE_HEIGHT = CART_LEVEL + CART_HEIGHT
	"""docstring for CartPole"""
	def __init__(self, config, figure = False, beta = 0.01):
		self.g = config["acceleration_of_gravity"]
		self.m_c = config["mass_of_cart"]
		self.m_p = config["mass_of_pole"]
		self.m = self.m_c + self.m_p

		self.t = config["update_time_interval"]

		# half length of pole
		self.hl = config["half_length_of_pole"]
		self.f = config["force"]
		self.beta = beta

		self.iniState()

		if figure:	
			self.iniFigure()
		
	
	def reset(self):
		self.iniState()
		self.setCart(self.x)
		self.setPole(self.x, self.theta)

	def iniState(self):

		self.v_theta = self.beta * np.random.random_sample()
		self.theta = self.beta * np.random.random_sample()
		self.x = self.beta * np.random.random_sample()
		self.v_x = self.beta * np.random.random_sample()

	def iniFigure(self):

		self.setCart(0.0)
		self.setPole(0, 0)
		self.fig = pl.figure()
		self.ax = self.fig.add_subplot(111, aspect='equal')
		self.ax.set_xlim([-3, 3])
		self.ax.set_ylim([0, 3])

		# self.ax.add_patch(self.cart)
		# self.ax.add_line(self.pole)
		self.draw()


	def getState(self):
		return {
			"x": self.x,
			"v_x": self.v_x,
			"theta": self.theta,
			"v_theta": self.v_theta
		}

	def update(self, direction = 1):



		acceleration_of_theta =((self.m * self.g * np.sin(self.theta) - 
			np.cos(self.theta) * (direction * self.f + self.m_p * self.hl * np.power(self.v_theta, 2) * np.sin(self.theta))) /
					 ((4/3) * self.m * self.hl - self.m_p * self.hl * np.power(np.cos(self.theta), 2)))

		acceleration_of_x = (direction * self.f + self.m_p * self.hl * (np.power(self.v_theta, 2) * np.sin(self.theta) - acceleration_of_theta * np.cos(self.theta))) / self.m;


		self.v_theta += acceleration_of_theta * self.t
		self.v_x += acceleration_of_x * self.t

		self.theta += self.v_theta * self.t
		self.x += self.v_x * self.t


		print(acceleration_of_theta, acceleration_of_x)

		self.setCart(self.x)
		self.setPole(self.x, self.theta)

	def setPole(self, x, theta):
		self.pole = lines.Line2D([x, x + np.cos(np.pi/2 - theta)], 
			[self.POLE_HEIGHT, self.POLE_HEIGHT + np.sin(np.pi/2 - theta)], color="red", lw=1.5)

	def setCart(self, x):
		self.cart = patches.Rectangle((x - self.CART_WIDTH / 2, self.CART_LEVEL), self.CART_WIDTH, self.CART_HEIGHT)


	def fakeUpdate(self):
		x = np.random.randn()
		self.cart = patches.Rectangle((x, 5.0), 1, 1)

	def draw(self):
		
		# self.cart.remove()
		self.ax.cla()
		self.ax.add_patch(self.cart)
		self.ax.add_line(self.pole)
		pl.pause(0.03)
		


# # point should be:
# # x_1 = (x, 0.2)
# # x_2 = (x + np.cos(np.pi/2 - self.theta), 0.2 + np.sin(np.pi/2 - self.theta))

# fig1 = pl.figure()
# ax1 = fig1.add_subplot(111, aspect='equal')


# def moveCart(x, y):
# 	cart =  patches.Rectangle(
#         (0.1, 0.1),   # (x,y)
#         0.5,          # width
#         0.5,          # height
#     )
# 	ax1.add_patch(cart)


# animation.FuncAnimation(fig1, )

# pl.pause(1)
# input("input something")
# pl.show()