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
sandstone = 2500                  # velocity in m/s
limestone = 3500                  # velocity in m/s
ny, nx = 500, 500                 # model size
time_steps = [150, 160, 170, 180] # snapshot of wave propagation (ms)
freq = 25                         # Frequency of the source in Hz 
dx = 4.0                          # Spatial sampling interval in meters 
dt = 0.004                        # Temporal sampling interval in secondiiis
output_folder = "image_out"

#-----------------------------------------------------------------------------------------#

source_location = torch.tensor([[[0, nx // 2]]]).to(device)

vp = sandstone * torch.ones(ny, nx)
vp[int(ny // 2):, :] = limestone
vp = vp.to(device)

#-----------------------------------------------------------------------------------------#

# Plot the wave propagation
U.plot_wave_propagation(vp, dx, dt, freq, time_steps, device, source_location, output_folder)

#-----------------------------------------------------------------------------------------#