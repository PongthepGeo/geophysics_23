#-----------------------------------------------------------------------------------------#
import os
import numpy as np
import matplotlib.pyplot as plt
import math
#-----------------------------------------------------------------------------------------#

data = np.load("data_out/densities_image.npy")
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
# NOTE Scenario 2: Removing the first layer (0-655 in axis-y) but keep gold
data_1 = data.copy()
data_1[:655] = np.where(data_1[:655] != 19300, 0, data_1[:655])
potentials_1 = compute_potentials(data_1)
print(f"finished scenario 2")
# NOTE Scenario 3: Removing up to the second layer (0-1360 in axis-y) but keep gold
data_2 = data.copy()
data_2[:1360] = np.where(data_2[:1360] != 19300, 0, data_2[:1360])
potentials_2 = compute_potentials(data_2)
print(f"finished scenario 3")
# NOTE Scenario 4: Removing all layers except for gold
data_3 = np.where(data != 19300, 0, data)
potentials_3 = compute_potentials(data_3)
print(f"finished scenario 4")

#-----------------------------------------------------------------------------------------#

plt.figure(figsize=(10, 6))
plt.plot(range(0, data.shape[1], 100), potentials_original, label="Original Data")
plt.plot(range(0, data_1.shape[1], 100), potentials_1, label="After Removing 0-655 Layer")
plt.plot(range(0, data_2.shape[1], 100), potentials_2, label="After Removing 0-1360 Layer")
plt.plot(range(0, data_3.shape[1], 100), potentials_3, label="Only Gold")
plt.xlabel("Gravity Station x-coordinate")
plt.ylabel("Total Potential")
plt.legend()
plt.title("Gravity Potential Profiles")
# plt.savefig("data_out/gravity_profiles.svg", format='svg', bbox_inches='tight', pad_inches=0, transparent=True, dpi=80)
plt.show()

#-----------------------------------------------------------------------------------------#