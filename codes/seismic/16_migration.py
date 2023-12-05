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
dtype = torch.float

#-----------------------------------------------------------------------------------------#

# NOTE Main define parameters
image_path = 'data/modeling/gold.png'
minimum_velocity = 1500
maximum_velocity = 4500
smooth = 40            # Smooth the velocity model, the higher the smoother (reduce scattering)
freq = 25               # Frequency of the source in Hz 
dx = 4.0                # Spatial sampling interval in meters 
dt = 0.004              # Temporal sampling interval in seconds
peak_time = 1.5 / freq
shot_interval = 10      # Every 10 pixel will allocate 1 shot
nt = 400
npy_folder = 'npy_folder'  # Load shot data from this folder
# NOTE Optimization parameters
optimizer_name = 'Adam'; lr=1e-4
loss_fn_name = 'MSELoss'
n_epochs = 1
# NOTE Output folder and save images
output_folder = 'image_out'
output_velocity_name = 'vp_3.svg'
output_migration_name = 'migrated_image.svg'

#-----------------------------------------------------------------------------------------#

# NOTE Image to velocity model conversion
img = Image.open(image_path)
width, height = img.size
img_resized = img.resize((width // 2, height // 2))
processor = C.Image2Velocity(img_resized, smooth)
vp_array = processor.plot_velocity(minimum_velocity, maximum_velocity, output_folder)
nx = vp_array.shape[1]
vp = torch.tensor(vp_array, dtype=dtype).to(device)
vp = torch.transpose(vp, 0, 1)  # Transpose the model
vp = vp.to(device)

#-----------------------------------------------------------------------------------------#

# NOTE Source parameters
n_shots = nx // shot_interval  # Number of shots based on model width 
n_sources_per_shot = 1
source_depth = 2
source_locations = torch.zeros(1, n_sources_per_shot, 2, dtype=dtype, device=device)  # Note the '1' for a single shot
source_locations[..., 1] = source_depth

#-----------------------------------------------------------------------------------------#

# NOTE Receiver parameters
d_receiver = 3  # Receiver interval (grid points)
n_receivers_per_shot = nx // d_receiver
receiver_depth = 0
receiver_locations = torch.zeros(1, n_receivers_per_shot, 2, dtype=dtype, device=device)
receiver_locations[..., 1] = receiver_depth
receiver_locations[0, :, 0] = torch.arange(0, nx, d_receiver).long()[:n_receivers_per_shot]

#-----------------------------------------------------------------------------------------#

# NOTE Source amplitudes
source_amplitudes = deepwave.wavelets.ricker(freq, nt, dt, peak_time).reshape(1, 1, -1).to(dtype=dtype, device=device)

#-----------------------------------------------------------------------------------------#

# NOTE Computing migration
inversion = C.Migration(vp, npy_folder, dtype, device, dx, dt, source_amplitudes, 
                        receiver_locations, freq)
inversion.setup_optimizer(optimizer_name=optimizer_name, lr=lr)
inversion.setup_loss_function(loss_fn_name=loss_fn_name)
inversion.run_inversion(n_epochs=n_epochs, shot_interval=shot_interval, n_shots=n_shots)
inversion.plot_migration(output_folder, output_migration_name, clip_percent=95, save=True)

#-----------------------------------------------------------------------------------------#