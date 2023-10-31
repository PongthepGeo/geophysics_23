#-----------------------------------------------------------------------------------------#
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
import numpy as np
np.seterr(all='ignore')
#-----------------------------------------------------------------------------------------#
import sys
sys.path.append('./Libs') 
import classes as C
# import utilities as U
#-----------------------------------------------------------------------------------------#
from PIL import Image
import torch
import deepwave
#-----------------------------------------------------------------------------------------#

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

#-----------------------------------------------------------------------------------------#

# NOTE Predine velocity model parameters
image_path = 'data/modeling/dome_fault.png'
minimum_velocity = 1500
maximum_velocity = 4000

#-----------------------------------------------------------------------------------------#

# NOTE Predine wave propagation parameters
time_steps = [50, 70, 140, 200] # snapshot of wave propagation (ms)
freq = 30                       # Frequency of the source in Hz 
dx = 4.0                        # Spatial sampling interval in meters 
dt = 0.004                      # Temporal sampling interval in seconds
output_folder = "image_out"

#-----------------------------------------------------------------------------------------#

# NOTE Import the image and convert it to velocity model
img = Image.open(image_path)
img_processor = C.ImageToVelocity(img)
velocity_array = img_processor.photo2velocity(minimum_velocity, maximum_velocity, output_folder)

#-----------------------------------------------------------------------------------------#

# NOTE Create velocity model and locate source
ny, nx = img_processor.img_arr.shape
source_location = torch.tensor([[[0, nx // 2]]]).to(device)
vp = torch.tensor(velocity_array, dtype=torch.float64).to(device)

#-----------------------------------------------------------------------------------------#

# NOTE Compute wave propagation and plot snapshots of wave propagation
img_processor.plot_wave_propagation(vp, dx, dt, freq, time_steps, device, source_location,
                                    output_folder)

#-----------------------------------------------------------------------------------------#