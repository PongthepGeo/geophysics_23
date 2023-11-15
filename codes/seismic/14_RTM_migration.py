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
#-----------------------------------------------------------------------------------------#

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

#-----------------------------------------------------------------------------------------#

# NOTE Predefine velocity model parameters
image_path = 'data/modeling/dome_fault.png'
minimum_velocity = 1500
maximum_velocity = 4000
smooth = 40
# NOTE Predefine source and receiver parameters
output_folder = "image_out"
freq = 15               # Frequency of the source in Hz 
dx = 4.0                # Spatial sampling interval in meters 
dt = 0.004              # Temporal sampling interval in seconds

#-----------------------------------------------------------------------------------------#

# NOTE Image to velocity model conversion
img = Image.open(image_path)
processor = C.Image2Velocity(img, smooth)
processor.plot_velocity(minimum_velocity, maximum_velocity, output_folder)



# #-----------------------------------------------------------------------------------------#

# # NOTE Create velocity model and locate source
# # ny, nx = img_processor.img_arr.shape
# # source_location = torch.tensor([[[0, nx // 2]]]).to(device)
# vp = torch.tensor(velocity_array, dtype=torch.float64).to(device)

# v_mig = torch.tensor(1/gaussian_filter(1/vp.numpy(), 40))
# ny = v_mig.shape[0]
# nx = v_mig.shape[1]

# v_mig = torch.transpose(v_mig, 0, 1)  # Transpose the model
# v_mig = v_mig.to(device)

# print(v_mig.shape)

#-----------------------------------------------------------------------------------------#
