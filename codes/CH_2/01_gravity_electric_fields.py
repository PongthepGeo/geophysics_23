#-----------------------------------------------------------------------------------------#
import utilities as U
import numpy as np
#-----------------------------------------------------------------------------------------#

# Define position of object mass
x0 = 5; y0 = 5
# Calculate distance from the Earth position (0, 0) to the object mass
r = np.sqrt(x0**2 + y0**2)
# Define constants
m = 1e24  # object mass in kg 
M = 5.972e24  # Earth mass in kg
G = 6.67430e-11  # gravitational constant in m^3 kg^-1 s^-2
# Normalize the x and y coordinates for the direction
rx = x0 / r
ry = y0 / r
# Calculate the gravitational force vectors
F = G * M * m / r**2
Fg_x = - F * rx  # gravitational force in x direction
Fg_y = - F * ry  # gravitational force in y direction

#-----------------------------------------------------------------------------------------#

U.Earth_field(x0, y0, Fg_x, Fg_y)

#-----------------------------------------------------------------------------------------#