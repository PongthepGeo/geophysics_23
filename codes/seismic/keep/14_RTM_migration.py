#-----------------------------------------------------------------------------------------#
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
import numpy as np
np.seterr(all='ignore')
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
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import os
#-----------------------------------------------------------------------------------------#

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

#-----------------------------------------------------------------------------------------#

# NOTE Predefine velocity model parameters
image_path = 'data/modeling/dome_fault.png'
minimum_velocity = 1500
maximum_velocity = 4000
smooth = 5
# NOTE Predefine source and receiver parameters
output_folder = "image_out"
freq = 25               # Frequency of the source in Hz 
dx = 4.0                # Spatial sampling interval in meters 
dt = 0.004              # Temporal sampling interval in seconds
peak_time = 1.5 / freq
nt = 700
# Directory for saving the output files
output_dir = 'npy_folder'
os.makedirs(output_dir, exist_ok=True)

#-----------------------------------------------------------------------------------------#

# NOTE Image to velocity model conversion
img = Image.open(image_path)
processor = C.Image2Velocity(img, smooth)
vp_array = processor.plot_velocity(minimum_velocity, maximum_velocity, output_folder)
nx = vp_array.shape[1]

#-----------------------------------------------------------------------------------------#

vp = torch.tensor(vp_array, dtype=torch.float64).to(device)
vp = torch.transpose(vp, 0, 1)  # Transpose the model
vp = vp.to(device)

#-----------------------------------------------------------------------------------------#

# NOTE Source location (one sourc at the top center of the model)
n_shots = 1
n_sources_per_shot = 1
d_source = 1  
current_source_position = int(nx // 2)  
source_depth = 2  
source_locations = torch.zeros(n_shots, n_sources_per_shot, 2, dtype=torch.long, device=device)
source_locations[..., 1] = source_depth
source_locations[:, 0, 0] = (torch.arange(n_shots) * d_source + current_source_position)
peak_time = 1.5 / freq  # The time at which the Ricker wavelet reaches its peak

#-----------------------------------------------------------------------------------------#

# NOTE Receiver location (approximately 10 meters receiver interval)
d_receiver = 3  # number of grid points between receivers, given dx = 4.0 m (approximating 10 meters by 12 meters)
n_receivers_per_shot = nx // d_receiver # approimately 10 meters receiver interval
receiver_depth = 0
receiver_locations = torch.zeros(n_shots, n_receivers_per_shot, 2, dtype=torch.long, device=device)
receiver_locations[..., 1] = receiver_depth
receiver_locations[:, :, 0] = torch.arange(0, nx, d_receiver).long()[:n_receivers_per_shot]

#-----------------------------------------------------------------------------------------#

pixel_number = source_locations[0, 0, 0].item()
print(f'computing shot at pixel number: {pixel_number}')
sw = C.LoopSeismicWavefield(freq, dt, peak_time, n_shots, n_sources_per_shot, device, vp, dx,
							source_locations, receiver_locations)
_, receiver_amplitudes = sw.loop_wavefield(nt)
np.save(os.path.join(output_dir, f'shot_pixel_{pixel_number:04d}.npy'), receiver_amplitudes.cpu().numpy())

#-----------------------------------------------------------------------------------------#