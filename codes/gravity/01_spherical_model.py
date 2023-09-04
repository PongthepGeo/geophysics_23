#-----------------------------------------------------------------------------------------#
import numpy as np
#-----------------------------------------------------------------------------------------#

G = 6.674*pow(10, -11)
gold_density = 19300 # kg/m3
sandstone_density = 2590 # kg/m3
density_contrast = gold_density - sandstone_density
gold_diameter = 100 # meters 
gold_depth = 200 # meters
distance_x = 500 # meters

#-----------------------------------------------------------------------------------------#

def gravity_sphere(G, density_contrast, R, z, x):
	x1 = density_contrast * ((4/3) * np.pi * pow(R, 3)) * z
	x2 = pow((pow(x, 2) + pow(z, 2)), 1.5)
	gz = G * (x1/x2)
	gal = gz*0.01 # unit gal
	mgal = gal*pow(10, 6) # unit gal
	return mgal

#-----------------------------------------------------------------------------------------#

# NOTE compute 1 observation point
mgal_gz = gravity_sphere(G, density_contrast, gold_diameter/2, gold_depth, distance_x)
print(f"gravity at {distance_x} meters from the origin is: {mgal_gz} mgal")

#-----------------------------------------------------------------------------------------#