#-----------------------------------------------------------------------------------------#
import sys
sys.path.append('./Libs') 
import classes as C
#-----------------------------------------------------------------------------------------#
from PIL import Image
import torch
import deepwave
#-----------------------------------------------------------------------------------------#

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

#-----------------------------------------------------------------------------------------#

# NOTE Predine velocity model parameters
image_path = 'data/modeling/gold.png'
minimum_velocity = 1500
maximum_velocity = 4500

#-----------------------------------------------------------------------------------------#

# NOTE Predine wave propagation parameters
time_steps = [50, 70, 140, 200] # snapshot of wave propagation (ms)
freq = 25                       # Frequency of the source in Hz 
dx = 4.0                        # Spatial sampling interval in meters 
dt = 0.004                      # Temporal sampling interval in seconds
# NOTE Output folder and save images
output_folder = 'image_out'
output_velocity_name = 'vp.png'
output_wave_name = 'wave.png'

#-----------------------------------------------------------------------------------------#

# NOTE Import the image and convert it to velocity model
img = Image.open(image_path)
img_processor = C.ImageToVelocity(img)
velocity_array = img_processor.photo2velocity(minimum_velocity, maximum_velocity,
                                              output_folder, output_velocity_name, save=True)

#-----------------------------------------------------------------------------------------#

# NOTE Create velocity model and locate source
ny, nx = img_processor.img_arr.shape
source_location = torch.tensor([[[0, nx // 2]]]).to(device)
vp = torch.tensor(velocity_array, dtype=torch.float64).to(device)

#-----------------------------------------------------------------------------------------#

# NOTE Compute wave propagation and plot snapshots of wave propagation
img_processor.plot_wave_propagation(vp, dx, dt, freq, time_steps, device, source_location,
                                    output_folder, output_wave_name, save=True)

#-----------------------------------------------------------------------------------------#