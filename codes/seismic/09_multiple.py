#-----------------------------------------------------------------------------------------#
import sys
sys.path.append('./Libs') 
import utilities as U
#-----------------------------------------------------------------------------------------#
import torch
#-----------------------------------------------------------------------------------------#

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

#-----------------------------------------------------------------------------------------#

# NOTE Create a velocity model	
sandstone = 2500  # velocity in m/s
salt = 4500       # velocity in m/s
ny, nx = 500, 500
time_steps = [50, 70, 140, 200] # snapshot of wave propagation (ms)
freq = 25                     # Frequency of the source in Hz 
dx = 4.0                     # Spatial sampling interval in meters 
dt = 0.004                    # Temporal sampling interval in seconds
output_folder = "image_out"

#-----------------------------------------------------------------------------------------#

source_location = torch.tensor([[[0, nx // 2]]]).to(device)

salt_end = ny // 3                 
sandstone_end = 2 * ny // 3      
vp = sandstone * torch.ones(ny, nx)
vp[:salt_end, :] = salt
vp[salt_end:sandstone_end, :] = sandstone
vp[sandstone_end:, :] = salt
vp = vp.to(device)

#-----------------------------------------------------------------------------------------#

# Plot the wave propagation
U.plot_wave_propagation(vp, dx, dt, freq, time_steps, device, source_location, output_folder)

#-----------------------------------------------------------------------------------------#
