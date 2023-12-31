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

# NOTE Predefine parameters
sandstone = 2500                 # velocity in m/s
ny, nx = 1000, 1000              # model size
time_steps = [80, 120, 140, 180] # snapshot of wave propagation (ms)
freq = 25                        # Frequency of the source in Hz 
dx = 4.0                         # Spatial sampling interval (distance between grid points) in meters 
dt = 0.004                       # Temporal sampling interval (time step) in seconds
output_folder = "image_out"

#-----------------------------------------------------------------------------------------#

# NOTE Create a source location
source_location = torch.tensor([[[ny // 2, nx // 2]]]).to(device)
# NOTE Create a velocity model	
vp = sandstone * torch.ones(ny, nx)
vp = torch.transpose(vp, 0, 1)  # Transpose the model
vp = vp.to(device)

#-----------------------------------------------------------------------------------------#

# NOTE Plot the wave propagation
U.plot_wave_propagation(vp, dx, dt, freq, time_steps, device, source_location, output_folder)

#-----------------------------------------------------------------------------------------#