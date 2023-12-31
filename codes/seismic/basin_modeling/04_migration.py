#-----------------------------------------------------------------------------------------#
import sys
sys.path.append('./Libs') 
import classes as C
#-----------------------------------------------------------------------------------------#
import numpy as np
from PIL import Image
import torch
import deepwave
#-----------------------------------------------------------------------------------------#

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

#-----------------------------------------------------------------------------------------#

# NOTE Main define parameters
image_path = 'models/basin_01.png'
minimum_velocity = 1500
maximum_velocity = 4500
smooth = 20             # Smooth the velocity model, the higher the smoother (reduce scattering)
freq = 35               # Frequency of the source in Hz 
dx = 4.0                # Spatial sampling interval in meters 
dt = 0.004              # Temporal sampling interval in seconds
peak_time = 1.5 / freq
shot_interval = 5      # Every 10 pixel will allocate 1 shot
nt = 700              # Number of time steps, how long wave propagates
npy_folder = 'npy_folder'  # Load shot data from this folder
# NOTE Optimization parameters
optimizer_name = 'AdamW'; lr=1e-9
loss_fn_name = 'CrossEntropyLoss'
n_epochs = 1
# NOTE Output folder and save images
output_folder = 'image_out'
output_velocity_name = 'vp.png'
output_migration_name = 'migrated_image_sm20.svg'

#-----------------------------------------------------------------------------------------#

# NOTE Image to velocity model conversion
img = Image.open(image_path)
processor = C.Image2Velocity(img, smooth)
vp_array = processor.plot_velocity(minimum_velocity, maximum_velocity, output_folder)
nx = vp_array.shape[1]
vp = torch.tensor(vp_array, dtype=torch.float32).to(device)
vp = torch.transpose(vp, 0, 1)  # Transpose the model
vp = vp.to(device)

#-----------------------------------------------------------------------------------------#

# NOTE Source parameters
n_shots = nx // shot_interval  # Number of shots based on model width 
n_sources_per_shot = 1
source_depth = 2
source_locations = torch.zeros(1, n_sources_per_shot, 2, dtype=torch.float32, device=device) # Note the '1' for a single shot
source_locations[..., 1] = source_depth

#-----------------------------------------------------------------------------------------#

# NOTE Receiver parameters
d_receiver = 2  # Receiver interval (grid points)
n_receivers_per_shot = nx // d_receiver
receiver_depth = 0
receiver_locations = torch.zeros(1, n_receivers_per_shot, 2, dtype=torch.float32, device=device)
receiver_locations[..., 1] = receiver_depth
receiver_locations[0, :, 0] = torch.arange(0, nx, d_receiver, dtype=torch.float32)[:n_receivers_per_shot]

#-----------------------------------------------------------------------------------------#

# NOTE Source amplitudes
source_amplitudes = deepwave.wavelets.ricker(freq, nt, dt, peak_time).reshape(1, 1, -1).to(dtype=torch.float32, device=device)

#-----------------------------------------------------------------------------------------#

# NOTE Computing migration
inversion = C.Migration(vp, npy_folder, device, dx, dt, source_amplitudes, receiver_locations, freq)
# inversion.setup_optimizer(optimizer_name=optimizer_name, lr=lr)
# inversion.setup_loss_function(loss_fn_name=loss_fn_name)
# inversion.run_inversion(n_epochs, shot_interval, n_shots)
inversion.plot_migration(output_folder, output_migration_name, clip_percent=95, save=True)

#-----------------------------------------------------------------------------------------#