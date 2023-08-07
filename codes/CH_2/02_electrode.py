#-----------------------------------------------------------------------------------------#
import utilities as U
import numpy as np
#-----------------------------------------------------------------------------------------#

# Grid size
size_x = 20
size_y = 20
# Create a 2D grid
grid_x, grid_y = np.meshgrid(np.linspace(-50, 50, size_x), np.linspace(-50, 50, size_y))
# Electrode position (center of the grid)
x_source = 0.0
y_source = 0.0
# Current and resistivity
I = 150.0
rho = 1.0

#-----------------------------------------------------------------------------------------#

# Calculate the potential for the entire grid
V = U.calculate_potential(grid_x, grid_y, x_source, y_source, I, rho)
U.plot_contours(grid_x, grid_y, V)

#-----------------------------------------------------------------------------------------#
