#-----------------------------------------------------------------------------------------#
import sys
sys.path.append('./Libs') 
import classes as C
import utilities as U
#-----------------------------------------------------------------------------------------#
import numpy as np
from PIL import Image
import torch
import deepwave
from deepwave import scalar
import os
from tqdm import tqdm
import shutil
#-----------------------------------------------------------------------------------------#

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)
dtype = torch.float

#-----------------------------------------------------------------------------------------#

# NOTE Predefine velocity model parameters
image_path = 'data/modeling/gold.png'
minimum_velocity = 1500
maximum_velocity = 4500
smooth = 5              # Smooth the velocity model, the higher the smoother (reduce scattering)
# NOTE Predefine source and receiver parameters
output_folder = "image_out"
freq = 25               # Frequency of the source in Hz 
dx = 4.0                # Spatial sampling interval in meters 
dt = 0.004              # Temporal sampling interval in seconds
peak_time = 1.5 / freq
shot_interval = 10      # Every 10 pixel will allocate 1 shot
nt = 400              # Number of time steps, how long wave propagates
# NOTE Output folder 
output_dir = 'npy_folder'

#-----------------------------------------------------------------------------------------#

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
    print(f"Removed {output_dir}")
os.makedirs(output_dir)

#-----------------------------------------------------------------------------------------#

# NOTE Image to velocity model conversion
img = Image.open(image_path)
width, height = img.size
img_resized = img.resize((width // 2, height // 2))
processor = C.Image2Velocity(img_resized, smooth)
vp_array = processor.plot_velocity(minimum_velocity, maximum_velocity, output_folder)
nx = vp_array.shape[1]

#-----------------------------------------------------------------------------------------#

# NOTE Create velocity model and locate source
vp = torch.tensor(vp_array, dtype=dtype).to(device)
vp = torch.transpose(vp, 0, 1)  # Transpose the model
vp = vp.to(device)

#-----------------------------------------------------------------------------------------#

# NOTE Source location (one sourc at the top center of the model)
n_shots = nx // shot_interval  # Number of shots based on model width and interval of 10 pixels
n_sources_per_shot = 1
source_depth = 2
source_locations = torch.zeros(1, n_sources_per_shot, 2, dtype=dtype, device=device)  # Note the '1' for a single shot
source_locations[..., 1] = source_depth

#-----------------------------------------------------------------------------------------#

# NOTE Receiver location (approximately 10 meters receiver interval)
d_receiver = 3  # Receiver interval (grid points)
n_receivers_per_shot = nx // d_receiver
receiver_depth = 0
receiver_locations = torch.zeros(1, n_receivers_per_shot, 2, dtype=dtype, device=device)
receiver_locations[..., 1] = receiver_depth
receiver_locations[0, :, 0] = torch.arange(0, nx, d_receiver).long()[:n_receivers_per_shot]

#-----------------------------------------------------------------------------------------#

# NOTE Compute wave propagation for each shot
for i in tqdm(range(n_shots), desc="Computing shots"):
    current_source_position = i * shot_interval
    if current_source_position >= nx:
        break
    source_locations[0, 0, 0] = current_source_position
    sw = C.LoopSeismicWavefield(freq, dt, peak_time, 1, n_sources_per_shot, dtype, device, vp, dx,
                                source_locations, receiver_locations)
    _, receiver_amplitudes = sw.loop_wavefield(nt)
    np.save(os.path.join(output_dir, f'shot_pixel_{current_source_position:04d}.npy'),
                         receiver_amplitudes.cpu().numpy())

#-----------------------------------------------------------------------------------------#