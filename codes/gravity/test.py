import numpy as np
import math
import matplotlib.pyplot as plt

G = 6.674*pow(10, -11)
density = 19300 # kg/m3
area = 77*71
gold_mass = area*density
m1 = 4500
m2 = 3490
varie_mass = [4500,3490,3500,4200,6000,8000,7000,1200]

station_1 = np.array([580, 0])
station_2 = np.array([856, 0])
station_3 = np.array([1150, 0])
station_4 = np.array([1434, 0])
station_5 = np.array([1718, 0])
station = [[580, 0],[856, 0],[1150, 0],[1434, 0],[1718, 0]]
point_masses = [[1005, 737], [1031, 784], [1109, 771], [997, 830], [1065, 835], [1169, 819], [1129, 874], [1040, 896]]

def distance(point1, point2):
	x1, y1 = point1
	x2, y2 = point2
	return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def potential(G, gold_mass, dis):
	return G*(gold_mass/dis)

# total_U = 0 # initialize the accumulator
total_U = np.zeros(len(station))

for i in range(len(station)):
	print('Station'+str(i+1))
	for j in range(len(varie_mass)):
		r = distance(station[i],point_masses[j])
		U = potential(G,varie_mass[j],r)
		# total_U += U
		total_U[i] += U
		print(f"point mass at: {point_masses[j]} distance: {r} meter give potential: {U} j/kg")
	# print(total_U)
	# total_U = 0

x = np.linspace(1, len(station), len(station))
plt.scatter(x, total_U)
plt.show()