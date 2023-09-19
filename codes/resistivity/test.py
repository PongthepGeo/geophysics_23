from discretize import TreeMesh
from discretize.utils import mkvc, refine_tree_xyz, active_from_xyz

from SimPEG.utils import model_builder
from SimPEG.utils.io_utils.io_utils_electromagnetics import write_dcip2d_ubc
from SimPEG import maps, data
from SimPEG.electromagnetics.static import resistivity as dc
from SimPEG.electromagnetics.static.utils.static_utils import (
    generate_dcip_sources_line,
    apparent_resistivity_from_voltage,
    plot_pseudosection,
)

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

try:
    from pymatsolver import Pardiso as Solver
except ImportError:
    from SimPEG import SolverLU as Solver

write_output = False
mpl.rcParams.update({"font.size": 16})
# sphinx_gallery_thumbnail_number = 3

x_topo, y_topo = np.meshgrid(
    np.linspace(-3000, 3000, 601), np.linspace(-3000, 3000, 101)
)
z_topo = 40.0 * np.sin(2 * np.pi * x_topo / 800) - 40.0
x_topo, y_topo, z_topo = mkvc(x_topo), mkvc(y_topo), mkvc(z_topo)
topo_xyz = np.c_[x_topo, y_topo, z_topo]

# Create 2D topography. Since our 3D topography only changes in the x direction,
# it is easy to define the 2D topography projected along the survey line. For
# arbitrary topography and for an arbitrary survey orientation, the user must
# define the 2D topography along the survey line.
topo_2d = np.unique(topo_xyz[:, [0, 2]], axis=0)

# Define survey line parameters
survey_type = "dipole-dipole"
dimension_type = "2D"
data_type = "volt"
end_locations = np.r_[-400.0, 400.0]
station_separation = 40.0
num_rx_per_src = 10

# Generate source list for DC survey line
source_list = generate_dcip_sources_line(
    survey_type,
    data_type,
    dimension_type,
    end_locations,
    topo_2d,
    num_rx_per_src,
    station_separation,
)

# Define survey
survey = dc.survey.Survey(source_list, survey_type=survey_type)

dh = 4  # base cell width
dom_width_x = 3200.0  # domain width x
dom_width_z = 2400.0  # domain width z
nbcx = 2 ** int(np.round(np.log(dom_width_x / dh) / np.log(2.0)))  # num. base cells x
nbcz = 2 ** int(np.round(np.log(dom_width_z / dh) / np.log(2.0)))  # num. base cells z

# Define the base mesh
hx = [(dh, nbcx)]
hz = [(dh, nbcz)]
mesh = TreeMesh([hx, hz], x0="CN")

# Mesh refinement based on topography
# mesh = refine_tree_xyz(
#     mesh,
#     topo_xyz[:, [0, 2]],
#     octree_levels=[0, 0, 4, 4],
#     method="surface",
#     finalize=False,
# )

mesh.refine_surface(topo_xyz[:, [0, 2]], level=4, finalize=False)

# Mesh refinement near transmitters and receivers. First we need to obtain the
# set of unique electrode locations.
electrode_locations = np.c_[
    survey.locations_a,
    survey.locations_b,
    survey.locations_m,
    survey.locations_n,
]

unique_locations = np.unique(
    np.reshape(electrode_locations, (4 * survey.nD, 2)), axis=0
)

# Use loop to refine one point at a time
for point in unique_locations:
    mesh.refine_points(point[None, :], level=4, finalize=False)

# Refine core mesh region
lower_left = [-600, -400]
upper_right = [600, 0]

# Modify this based on the correct function signature
mesh.refine_bounding_box(lower_left, upper_right, octree_levels=[0, 0, 2, 8], finalize=False)

mesh.finalize()

