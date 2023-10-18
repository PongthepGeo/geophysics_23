#-----------------------------------------------------------------------------------------#
import torch
import matplotlib.pyplot as plt
import deepwave
from deepwave import scalar
#-----------------------------------------------------------------------------------------#

# NOTE Create a velocity model	
ny, nx = 100, 100
vp = 2500 * torch.ones(ny, nx)

#-----------------------------------------------------------------------------------------#

# NOTE Create a source location	
source_location = torch.tensor([[[ny // 2, nx // 2]]])
# NOTE Create a source amplitude	
freq = 25 # Hz
nt = 80 # Number of time steps, the longer nt, the longer wave propagation		
dt = 0.002 # Time step (second), the smaller dt, the longer wave propagation
peak_time = 1.5 / freq
dx = 8.0 # Spatial step (meter)

#-----------------------------------------------------------------------------------------#

# NOTE Create a wavefield
out = scalar(vp, dx, dt,
			 source_amplitudes=(
			 deepwave.wavelets.ricker(freq, nt, dt, peak_time).reshape(1, 1, -1)),
			 source_locations=source_location,
			 accuracy=2,
			 pml_freq=freq)

#-----------------------------------------------------------------------------------------#

print(out.shape)
# NOTE Plot snapshots of the wavefield
# _, ax = plt.subplots(2, figsize=(3.5, 5), sharey=True, sharex=True)
# ax[0].imshow(out[0][0, :, :], cmap='gray')
# ax[0].set_title('Wavefield y')
# ax[0].set_xticks([])
# ax[0].set_yticks([])
# ax[1].imshow(out[1][0, :, :], cmap='gray')
# ax[1].set_title('Wavefield x')
# ax[1].set_xticks([])
# ax[1].set_yticks([])
# plt.tight_layout()
# plt.savefig('modified_example_elastic_wavefield.jpg')