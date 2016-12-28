
########################################################################################################
from pynn import nnsimulation 

config = {
	"acceleration_of_gravity": 9.8,
	"mass_of_cart": 1,
	"mass_of_pole": 0.1,
	"update_time_interval": 0.2,
	"half_length_of_pole": 0.5,
	"force": 1
}

cp = nnsimulation.CartPole(config)


while True:
	cp.draw()
	cp.update()

	input("input>>")
########################################################################################################
# from matplotlib.figure import Figure                       
# from matplotlib.axes import Axes                           
# from matplotlib.lines import Line2D                        
# from matplotlib.backends.backend_agg import FigureCanvasAgg

# fig = Figure(figsize=[4,4])                                
# ax = Axes(fig, [.1,.1,.8,.8])                              
# fig.add_axes(ax)                                           
# l = Line2D([0,1],[0,1])                                    
# ax.add_line(l)                                             

# canvas = FigureCanvasAgg(fig)                              
# canvas.print_figure("line_ex.png") 