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
salt = 4500  # background velocity in m/s
box_velocity = 1500  # velocity in m/s for the box
ny, nx = 500, 500
box_start_x, box_end_x = 200, 300
box_start_y, box_end_y = 300, 400
time_steps = [85, 95, 105, 115] # snapshots of wave propagation (ms)
freq = 25  # Frequency of the source in Hz 
dx = 4.0  # Spatial sampling interval in meters 
dt = 0.004  # Temporal sampling interval in seconds
output_folder = "image_out"

#-----------------------------------------------------------------------------------------#

source_location = torch.tensor([[[0, nx // 2]]]).to(device)

vp = salt * torch.ones(ny, nx)
vp[box_start_y:box_end_y, box_start_x:box_end_x] = box_velocity
vp = vp.to(device)

#-----------------------------------------------------------------------------------------#

# Plot the wave propagation
U.box(vp, dx, dt, freq, time_steps, device, source_location,
      box_start_x, box_start_y, box_end_x, box_end_y,
	output_folder)

#-----------------------------------------------------------------------------------------#