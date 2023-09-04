import numpy as np
import pygimli as pg
from pygimli.meshtools import createCircle, createWorld, createLine
from pygimli.physics.gravimetry import (solveGravimetry)

import matplotlib.pyplot as plt

rho_1 = 2600.0
rho_2 = 2600.0
rho_3 = 2600.0

x = np.arange(-100, 100, 1)
pnts = np.zeros((len(x), 2))
pnts[:, 0] = x

world = createWorld(start=[-100, 0], end=[100, -50], marker=0, worldMarker=True)
cir_1 = createCircle(pos=[-65, -35.], radius = [25, 8], marker=1, boundaryMarker=10)
cir_2 = createCircle(pos=[25, -20.], radius = [35, 2], marker=2, boundaryMarker=10)
cir_3 = createCircle(pos=[75, -25.], radius = [15, 1], marker=3, boundaryMarker=10)
# geom_1 = world + cir_1 
# geom_2 = world + cir_2
# geom_3 = world + cir_3
geom_1 = cir_1 
geom_2 = cir_2
geom_3 = cir_3

# geom = world + cir_1 + cir_2 + cir_3
# pg.show(geom)

g, gz_1 = solveGravimetry(geom_1, rho_1, pnts, complete=True)
g, gz_2 = solveGravimetry(geom_2, rho_2, pnts, complete=True)
g, gz_3 = solveGravimetry(geom_3, rho_3, pnts, complete=True)
# g, gz_4 = solveGravimetry(world, rho_3, pnts, complete=True)
# # g, gz = solveGravimetry(geom, rhomap, pnts, complete=True)
# # print(gz)
# gz = gz_1 + gz_2 + gz_3

# plt.plot(x, g[:, 2], marker='o', linestyle='', c="cyan")
# plt.plot(x, gz[:, 2], marker='o', linestyle='',
#             label=r'Won & Bevis: $\frac{\partial^2 u}{\partial z,z}$', c="blue")
plt.plot(x, gz_1[:, 2], marker='x', linestyle='',
            label=r'Won & Bevis: $\frac{\partial^2 u}{\partial z,z}$', c="red")
plt.plot(x, gz_2[:, 2], marker='x', linestyle='',
            label=r'Won & Bevis: $\frac{\partial^2 u}{\partial z,z}$', c="green")
plt.plot(x, gz_3[:, 2], marker='x', linestyle='',
            label=r'Won & Bevis: $\frac{\partial^2 u}{\partial z,z}$', c="black")
plt.savefig('image_out/Q1.4_gravity' + '.svg', format='svg', bbox_inches='tight', transparent=True, pad_inches=0)
plt.show()