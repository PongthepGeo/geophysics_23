#-----------------------------------------------------------------------------------------#
import os
import numpy as np
import matplotlib.pyplot as plt
import math
#-----------------------------------------------------------------------------------------#

data = np.load("data_out/exercise.npy")
# data = data[1000:, :]
# plt.imshow(data)
# plt.show()
data = np.flipud(data)
output_dir = "data_out"
G = 6.674 * pow(10, -11)
step = 10  # Step size

#-----------------------------------------------------------------------------------------#

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

#-----------------------------------------------------------------------------------------#

def distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def potential(G, mass, dis):
    return G * (mass / dis)

def compute_potentials(data_input):
    num_gravity_stations = len(range(0, data_input.shape[1], 100))
    gravitational_potentials = np.zeros(num_gravity_stations)
    for idx, i in enumerate(range(0, data_input.shape[1], 100)):
        total_potential = 0
        for y in range(0, data_input.shape[0], step):
            for x in range(0, data_input.shape[1], step):
                mass = data_input[y, x]
                dist = distance((i, 0), (x, y))
                if dist != 0:
                    total_potential += potential(G, mass, dist)
        gravitational_potentials[idx] = total_potential
    return gravitational_potentials

#-----------------------------------------------------------------------------------------#

# NOTE Scenario 1: Original data
potentials_original = compute_potentials(data)
print(f"finished scenario 1")

#-----------------------------------------------------------------------------------------#

plt.figure(figsize=(10, 6))
plt.plot(range(0, data.shape[1], 100), potentials_original, label="Original Data")
plt.xlabel("Gravity Station x-coordinate")
plt.ylabel("Total Potential")
plt.legend()
plt.title("Gravity Potential Profiles")
plt.savefig("data_out/ex_G.svg", format='svg', bbox_inches='tight', pad_inches=0, transparent=True, dpi=80)
plt.show()

#-----------------------------------------------------------------------------------------#