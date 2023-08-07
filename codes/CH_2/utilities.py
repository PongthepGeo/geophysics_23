#-----------------------------------------------------------------------------------------#
import matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np
#-----------------------------------------------------------------------------------------#
params = {
	'savefig.dpi': 300,  
	'figure.dpi' : 300,
	'axes.labelsize':10,  
	'axes.titlesize':10,
	'axes.titleweight': 'bold',
	'legend.fontsize': 8,
	'xtick.labelsize':8,
	'ytick.labelsize':8,
	'font.family': 'serif',
}
matplotlib.rcParams.update(params)
#-----------------------------------------------------------------------------------------#

if not os.path.exists('save_figures'):
	os.makedirs('save_figures')

#-----------------------------------------------------------------------------------------#

def Earth_field(x0, y0, Fg_x, Fg_y):
	fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
	plt.scatter(0, 0, color='blue', s=500)  # position of the Earth
	plt.scatter(x0, y0, color='red')  # position of the object mass
	plt.quiver(x0, y0, Fg_x, Fg_y, color='green')
	plt.xlim(-10, 10)
	plt.ylim(-10, 10)
	plt.gca().set_aspect('equal', adjustable='box')
	plt.title('Gravitational Force Vector at the Position of the Point Mass')
	plt.savefig('save_figures/' + 'resis_ex_03' + '.webp', format='webp', dpi=100,
				bbox_inches='tight', transparent=False, pad_inches=0)
	plt.show()

def calculate_potential(x_p, y_p, x_source, y_source, I, rho):
    # Calculate distance from point to source
    r = np.sqrt((x_p - x_source)**2 + (y_p - y_source)**2)
    # If r is zero, set potential to a very high value (e.g. infinity)
    r[r == 0] = np.inf
    # Use the formula to calculate the potential
    V = I * rho / (2 * np.pi * r)
    return V

def plot_contours(grid_x, grid_y, V):
	fig, ax = plt.subplots(figsize=(10, 10))  # Create a square figure
	contour = ax.contourf(grid_x, grid_y, V, levels=10, cmap='jet')
	plt.colorbar(contour, label='Potential')
	plt.title('Electric Potential Distribution')
	ax.set_aspect('equal')  # Set the aspect ratio of the plot to 1
	plt.savefig('save_figures/' + 'resis_ex_04' + '.webp', format='webp', dpi=100,
				bbox_inches='tight', transparent=False, pad_inches=0)
	plt.show()