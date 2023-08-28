#-----------------------------------------------------------------------------------------#
import numpy as np
import math
import matplotlib.pyplot as plt
#-----------------------------------------------------------------------------------------#

G = 6.674*pow(10, -11)
density = 19300 # kg/m3
area = 77*71
gold_mass = area*density

#-----------------------------------------------------------------------------------------#

station_1 = np.array([580, 0])
station_2 = np.array([856, 0])
station_3 = np.array([1150, 0])
station_4 = np.array([1434, 0])
station_5 = np.array([1718, 0])

point_masses = np.array([
    					[1005, 737],
                		[1031, 784],
                		[1109, 771],
                		[997, 830],
                		[1065, 835],
                		[1169, 819],
                		[1129, 874],
                		[1040, 896]
                  		])

#-----------------------------------------------------------------------------------------#

def distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def potential(G, gold_mass, dis):
	return G*(gold_mass/dis)

#-----------------------------------------------------------------------------------------#

# NOTE compute potential at station 1

total_U = 0  # initialize the accumulator
for point_mass in point_masses:
	dis = distance(station_1, point_mass)
	U = potential(G, gold_mass, dis) 
	print(f"point mass at: {point_mass} distance: {dis} meter give potential: {U} j/kg")
	total_U += U  # accumulate the potential
print(total_U)

#-----------------------------------------------------------------------------------------#