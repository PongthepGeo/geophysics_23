#-----------------------------------------------------------------------------------------#
import torch
import matplotlib.pyplot as plt
import deepwave
from deepwave import scalar
import numpy as np
import warnings
#-----------------------------------------------------------------------------------------#
warnings.filterwarnings("ignore", category=UserWarning)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
#-----------------------------------------------------------------------------------------#

ny = 2301; nx = 751; dx = 4.0

vp = 1500 * torch.ones(ny, nx)
layer_boundary = ny // 2  # Create a layer boundary in the middle of the y-axis
vp = torch.ones(ny, nx)  # Creating vp tensor
vp[:layer_boundary, :] = 1500  # Top layer
vp[layer_boundary:, :] = 4000  # Bottom layer
# Rotating the tensor 90 degrees
vp_rotated = torch.rot90(vp, k=1, dims=(0, 1))

#-----------------------------------------------------------------------------------------#

# NOTE in case for QC input velocity
vp_rotated = vp_rotated.cpu().numpy()
plt.imshow(vp_rotated, cmap='jet', vmin=1500, vmax=4000)  
plt.show()
