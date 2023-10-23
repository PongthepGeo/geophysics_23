#-----------------------------------------------------------------------------------------#
import sys
sys.path.append('./Libs') 
import utilities as U
#-----------------------------------------------------------------------------------------#
import torch
import deepwave
from deepwave import scalar
#-----------------------------------------------------------------------------------------#

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

#-----------------------------------------------------------------------------------------#

# NOTE Predefine parameters
sandstone = 2500        # velocity in m/s
limestone = 3700        # velocity in m/s
ny, nx = 1000, 1000     # model size
limestone_start = 500   # depth
time_steps = [250, 450] # snapshot of wave propagation (ms)
freq = 25               # Frequency of the source in Hz 
dx = 4.0                # Spatial sampling interval in meters 
dt = 0.004              # Temporal sampling interval in seconds
peak_time = 1.5 / freq  # The time at which the Ricker wavelet reaches its peak
output_folder = "image_out"

#-----------------------------------------------------------------------------------------#

# NOTE Create a velocity model
vp = sandstone * torch.ones(ny, nx)
vp[limestone_start:, :] = limestone
vp = torch.transpose(vp, 0, 1)  # Transpose the model
vp = vp.to(device)

#-----------------------------------------------------------------------------------------#

# NOTE Source location (one sourc at the top center of the model)
n_shots = 1
n_sources_per_shot = 1
d_source = 1  
first_source = int(nx // 2)  
source_depth = 2  
source_locations = torch.zeros(n_shots, n_sources_per_shot, 2, dtype=torch.long, device=device)
source_locations[..., 1] = source_depth
source_locations[:, 0, 0] = (torch.arange(n_shots) * d_source + first_source)

#-----------------------------------------------------------------------------------------#

# NOTE Receiver location (approximately 10 meters receiver interval)
d_receiver = 3  # number of grid points between receivers, given dx = 4.0 m (approximating 10 meters by 12 meters)
n_receivers_per_shot = nx // d_receiver # approimately 10 meters receiver interval
receiver_depth = 0
receiver_locations = torch.zeros(n_shots, n_receivers_per_shot, 2, dtype=torch.long, device=device)
receiver_locations[..., 1] = receiver_depth
receiver_locations[:, :, 0] = torch.arange(0, nx, d_receiver).long()[:n_receivers_per_shot]

#-----------------------------------------------------------------------------------------#

# NOTE Compute and plot the wave propagation plus the receivers	
U.plot_receivers(freq, dt, peak_time, n_shots, n_sources_per_shot, device, vp, dx,
				 source_locations, receiver_locations, time_steps, limestone_start, output_folder)

#-----------------------------------------------------------------------------------------#

