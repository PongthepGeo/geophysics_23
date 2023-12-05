#-----------------------------------------------------------------------------------------#
import sys
sys.path.append('./Libs') 
import classes as C
import utilities as U
#-----------------------------------------------------------------------------------------#
from PIL import Image
import torch
import deepwave
from deepwave import scalar
import os
#-----------------------------------------------------------------------------------------#

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

#-----------------------------------------------------------------------------------------#

# NOTE Predefine velocity model parameters
image_path = 'models/basin_01.png'
minimum_velocity = 1500
maximum_velocity = 4500
# NOTE Predefine source and receiver parameters
time_steps = [250, 700] # snapshot of wave propagation (ms)
freq = 35               # Frequency of the source in Hz 
peak_time = 1.5 / freq  # The time at which the Ricker wavelet reaches its peak
dx = 4.0                # Spatial sampling interval in meters 
dt = 0.004              # Temporal sampling interval in seconds
# NOTE Output folder and save images
output_folder = 'image_out'
output_velocity_name = 'vp.png'
output_receiver_name = 'receiver.png'

#-----------------------------------------------------------------------------------------#

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

#-----------------------------------------------------------------------------------------#

# NOTE Image to velocity model conversion
img = Image.open(image_path)
img_processor = C.ImageToVelocity(img)
velocity_array = img_processor.photo2velocity(minimum_velocity, maximum_velocity,
                                              output_folder, output_velocity_name, save=True)

#-----------------------------------------------------------------------------------------#

# NOTE Create velocity model and locate source
ny, nx = img_processor.img_arr.shape
vp = torch.tensor(velocity_array, dtype=torch.float32).to(device)
vp = torch.transpose(vp, 0, 1)  # Transpose the model
vp = vp.to(device)

#-----------------------------------------------------------------------------------------#

# NOTE Source location (one sourc at the top center of the model)
n_shots = 1
n_sources_per_shot = 1
d_source = 1  
first_source = int(nx // 2)  
source_depth = 2  
source_locations = torch.zeros(n_shots, n_sources_per_shot, 2, dtype=torch.float32, device=device)
source_locations[..., 1] = source_depth
source_locations[:, 0, 0] = (torch.arange(n_shots) * d_source + first_source)

# #-----------------------------------------------------------------------------------------#

# NOTE Receiver location (approximately 10 meters receiver interval)
d_receiver = 2  
n_receivers_per_shot = nx // d_receiver 
receiver_depth = 0
receiver_locations = torch.zeros(n_shots, n_receivers_per_shot, 2, dtype=torch.float32, device=device)
receiver_locations[..., 1] = receiver_depth
receiver_locations[:, :, 0] = torch.arange(0, nx, d_receiver).long()[:n_receivers_per_shot]

#-----------------------------------------------------------------------------------------#

seismic_wavefield = C.SeismicWavefield(freq, dt, peak_time, n_shots, n_sources_per_shot,
                                       device, vp, dx, source_locations, receiver_locations,
                                       time_steps)
seismic_wavefield.plot_receivers(output_folder, output_receiver_name, save=True)

#-----------------------------------------------------------------------------------------#