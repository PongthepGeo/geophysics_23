#-----------------------------Code12-----------------------------------------------------#
# import sys
# sys.path.append('/content/drive/MyDrive/Code/Libs2/classes.py')
# import classes as C
#-----------------------------------------------------------------------------------------#
from PIL import Image
import torch
import deepwave
#-----------------------------------------------------------------------------------------#
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)
#-----------------------------------------------------------------------------------------#

#-----------------------------------------------------------------------------------------#
import sys
sys.path.append('./Libs') 
import utilities as U
#-----------------------------------------------------------------------------------------#
import numpy as np
import matplotlib.pyplot as plt
import deepwave
from deepwave import scalar
import torch
from PIL import Image 
from scipy.ndimage import gaussian_filter
import os
from tqdm import tqdm
from deepwave import scalar_born
#-----------------------------------------------------------------------------------------#
import matplotlib
params = {
	'savefig.dpi': 300,  
	'figure.dpi' : 300,
	'axes.labelsize':12,  
	'axes.titlesize':12,
	'axes.titleweight': 'bold',
	'legend.fontsize': 10,
	'xtick.labelsize':10,
	'ytick.labelsize':10,
	'font.family': 'serif',
	'font.serif': 'Times New Roman'
}
matplotlib.rcParams.update(params)
#-----------------------------------------------------------------------------------------#

class ImageToVelocity:
	def __init__(self, img_arr):
		self.img_arr = np.array(img_arr)
	def photo2velocity(self, min_velocity, max_velocity, output_folder, output_image_name, save=False):
		print("...Creating Velocity Model...")
		self.img_arr = U.rgba_to_grayscale(self.img_arr)
		self.img_arr = U.normalize_data(self.img_arr, min_velocity, max_velocity)
		fig = plt.figure(figsize=(10, 8))
		plt.imshow(self.img_arr, cmap='gray')
		plt.title("Velocity Model")
		cbar = plt.colorbar()
		cbar.set_label('Velocity (m/s)')  # Add title to colorbar
		if save:
			save_name = output_folder + "/" + output_image_name
			plt.savefig(save_name, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
			print("Velocity model saved as:", save_name)
		plt.show()
		return self.img_arr

	def get_wavefield(self, vp, dx, dt, freq, nt, device, source_location):
		peak_time = 1.5 / freq 
		wavefields = scalar(vp, dx, dt,            
							source_amplitudes = deepwave.wavelets.ricker(freq, nt, dt, peak_time).reshape(1, 1, -1).to(dtype=torch.float64, device=device),
							source_locations = source_location,  
							accuracy = 8,    
							pml_freq = freq) 
		return wavefields

	def plot_wave_propagation(self, vp, dx, dt, freq, time_steps, device, source_location,
		output_folder, output_image_name, save=False):
		print("...Time Step Wavefield...")
		plt.figure()
		wavefields = [self.get_wavefield(vp, dx, dt, freq, nt, device, source_location) for nt in time_steps]
		pml_thickness = 20
		source_y = (source_location[0, 0, 0] + pml_thickness).item()
		source_x = (source_location[0, 0, 1] + pml_thickness).item()
		for idx, (wavefield, nt) in enumerate(zip(wavefields, time_steps), 1):
			plt.subplot(2, 2, idx)
			wave_data = wavefield[0][0, :, :].cpu().numpy() # extract array from tensor and move to CPU
			max_num, min_num = U.clip(wave_data, 100)
			plt.imshow(wave_data, cmap='gray', vmin=min_num, vmax=max_num)
			plt.scatter(source_x, source_y, c='blue', s=50)  # Plot blue dot at source location
			plt.xlabel('X Distance (m)')
			plt.ylabel('Y Distance (m)')
			plt.title(f"Time Step: {nt} ms")
		plt.subplots_adjust(wspace=0.1, hspace=0.6)  
		if save:
			save_name = output_folder + "/" + output_image_name
			plt.savefig(save_name, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
			print("Wave propagation saved as:", save_name)
		plt.show()

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
img_processor = ImageToVelocity(img)
velocity_array = img_processor.photo2velocity(minimum_velocity, maximum_velocity,output_folder, output_velocity_name,save = True)
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