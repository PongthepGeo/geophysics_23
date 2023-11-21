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

	def photo2velocity(self, min_velocity, max_velocity, output_folder):
		print("...Creating Velocity Model...")
		self.img_arr = U.rgba_to_grayscale(self.img_arr)
		self.img_arr = U.normalize_data(self.img_arr, min_velocity, max_velocity)
		fig = plt.figure(figsize=(10, 8))
		plt.imshow(self.img_arr, cmap='gray')
		plt.title("Velocity Model")
		cbar = plt.colorbar()
		cbar.set_label('Velocity (m/s)')  # Add title to colorbar
		# plt.savefig(output_folder + "/multiple.svg", format='svg', bbox_inches='tight', pad_inches=0, transparent=True)
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

	def plot_wave_propagation(self, vp, dx, dt, freq, time_steps, device,
							  source_location, output_folder):
		print("...Time Step Wavefield...")
		plt.figure()
		wavefields = [self.get_wavefield(vp, dx, dt, freq, nt, device,
										 source_location) for nt in time_steps]
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
		# plt.savefig(output_folder + "/complex_02.svg", format='svg', bbox_inches='tight', pad_inches=0, transparent=True)
		plt.show()

#-----------------------------------------------------------------------------------------#

class SeismicWavefield:
	def __init__(self, freq, dt, peak_time, n_shots, n_sources_per_shot, device, vp, dx, source_locations,
				 receiver_locations, time_steps, output_folder):
		self.freq = freq
		self.dt = dt
		self.peak_time = peak_time
		self.n_shots = n_shots
		self.n_sources_per_shot = n_sources_per_shot
		self.device = device
		self.vp = vp
		self.dx = dx
		self.source_locations = source_locations
		self.receiver_locations = receiver_locations
		self.time_steps = time_steps
		self.output_folder = output_folder
	
	def wavefield(self, nt):
		source_amplitudes = deepwave.wavelets.ricker(self.freq, nt, self.dt, self.peak_time).reshape(1, 1, -1).to(dtype=torch.float64, device=self.device)
		outputs = scalar(self.vp, self.dx, self.dt,
						 source_amplitudes=source_amplitudes,
						 source_locations=self.source_locations,
						 receiver_locations=self.receiver_locations,
						 accuracy=8,
						 pml_width=[40, 40, 40, 40],
						 pml_freq=self.freq)
		wavefields, receiver_amplitudes = outputs[0], outputs[-1]  
		return wavefields, receiver_amplitudes

	def plot_receivers(self):
		plt.figure()
		for i, nt in enumerate(self.time_steps):
			wave_propagation, receiver_data = self.wavefield(nt)
			wave_propagation = wave_propagation[0, :, :].cpu().numpy().T
			receiver_data = receiver_data[0].cpu().numpy().T
			# NOTE Wave Propagation
			plt.subplot(2, 2, 2*i + 1)
			y_max_wp = wave_propagation.shape[0] * self.dx * 0.001
			x_max_wp = wave_propagation.shape[1] * self.dx * 0.001
			max_wp, min_wp = U.clip(wave_propagation, 100)
			plt.imshow(wave_propagation, aspect='auto', cmap='gray', origin='upper',
					   extent=[0, x_max_wp, y_max_wp, 0], vmin=min_wp, vmax=max_wp)
			plt.title(f"Wave Propagation: {nt*0.001} s")
			plt.xlabel('Distance (km)')
			plt.ylabel('Depth (km)')
			# NOTE Receiver Data
			plt.subplot(2, 2, 2*i + 2)
			nt_seconds = nt * 0.001
			max_rd, min_rd = U.clip(receiver_data, 99)
			plt.imshow(receiver_data, aspect='auto', cmap='gray', origin='upper',
					   extent=[0, x_max_wp, nt_seconds, 0], vmin=min_rd, vmax=max_rd)
			plt.xlabel('Receiver Position (km)')
			plt.ylabel('Time (sec)')
			plt.title('Receiver')
		plt.subplots_adjust(wspace=0.4, hspace=0.6)
		plt.savefig(self.output_folder + "/receivers_complex.svg", format='svg', bbox_inches='tight', pad_inches=0, transparent=True)
		plt.show()

#-----------------------------------------------------------------------------------------#

class Image2Velocity:
	def __init__(self, img_array, sigma):
		self.img_array = np.array(img_array)
		self.sigma = sigma

	def plot_velocity(self, min_velocity, max_velocity, output_folder):
		print("...Creating Velocity Model...")
		self.img_array = U.rgba_to_grayscale(self.img_array)
		self.img_array = U.normalize_data(self.img_array, min_velocity, max_velocity)
		original_img_array = self.img_array.copy()
		self.img_array = gaussian_filter(self.img_array, sigma=self.sigma)
		fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), gridspec_kw={'wspace': 0.1})
		img1 = axes[0].imshow(original_img_array, cmap='rainbow')
		axes[0].set_xlabel('Distance-X (pixel)')
		axes[0].set_ylabel('Depth (pixel)')
		axes[0].set_title('Original Velocity Model')
		img2 = axes[1].imshow(self.img_array, cmap='rainbow')
		axes[1].set_xlabel('Distance-X (pixel)')
		axes[1].set_ylabel('Depth (pixel)')
		axes[1].set_title('Smoothed Velocity Model')
		cbar = fig.colorbar(img2, ax=axes.ravel().tolist(), orientation='horizontal', aspect=50)
		cbar.set_label('velocity (m/s)')
		plt.show()
		return self.img_array

#-----------------------------------------------------------------------------------------#

class LoopSeismicWavefield:
	def __init__(self, freq, dt, peak_time, n_shots, n_sources_per_shot, device, vp, dx,
				 source_locations, receiver_locations):
		self.freq = freq
		self.dt = dt
		self.peak_time = peak_time
		self.n_shots = n_shots
		self.n_sources_per_shot = n_sources_per_shot
		self.device = device
		self.vp = vp
		self.dx = dx
		self.source_locations = source_locations
		self.receiver_locations = receiver_locations
	
	def loop_wavefield(self, nt):
		source_amplitudes = deepwave.wavelets.ricker(self.freq, nt, self.dt, self.peak_time).reshape(1, 1, -1).to(dtype=torch.float64, device=self.device)
		outputs = scalar(self.vp, self.dx, self.dt,
						 source_amplitudes=source_amplitudes,
						 source_locations=self.source_locations,
						 receiver_locations=self.receiver_locations,
						 accuracy=8,
						 pml_width=[40, 40, 40, 40],
						 pml_freq=self.freq)
		wavefields, receiver_amplitudes = outputs[0], outputs[-1]  
		return wavefields, receiver_amplitudes

#-----------------------------------------------------------------------------------------#

class Migration:
	def __init__(self, vp, npy_folder, device, dx, dt, source_amplitudes, receiver_locations, freq):
		self.vp = vp
		self.npy_folder = npy_folder
		self.device = device
		self.dx = dx
		self.dt = dt
		self.freq = freq
		self.source_amplitudes = source_amplitudes
		self.receiver_locations = receiver_locations
		self.scatter = torch.zeros_like(vp, requires_grad=True)

	def setup_optimizer(self, optimizer_name='Adam', lr=1e-4):
		if optimizer_name == 'SGD':
			self.optimizer = torch.optim.SGD([self.scatter], lr=lr)
		elif optimizer_name == 'RMSprop':
			self.optimizer = torch.optim.RMSprop([self.scatter], lr=lr)
		elif optimizer_name == 'Adagrad':
			self.optimizer = torch.optim.Adagrad([self.scatter], lr=lr)
		elif optimizer_name == 'AdamW':
			self.optimizer = torch.optim.AdamW([self.scatter], lr=lr)
		else:
			self.optimizer = torch.optim.Adam([self.scatter], lr=lr)

	def setup_loss_function(self, loss_fn_name='MSELoss'):
		if loss_fn_name == 'CrossEntropyLoss':
			self.loss_fn = torch.nn.CrossEntropyLoss()
		elif loss_fn_name == 'BCEWithLogitsLoss':
			self.loss_fn = torch.nn.BCEWithLogitsLoss()
		elif loss_fn_name == 'NLLLoss':
			self.loss_fn = torch.nn.NLLLoss()
		elif loss_fn_name == 'L1Loss':
			self.loss_fn = torch.nn.L1Loss()
		else:
			self.loss_fn = torch.nn.MSELoss()

	def run_inversion(self, n_epochs, shot_interval, n_shots):
		observed_scatter_files = [
			os.path.join(self.npy_folder, f'shot_pixel_{i * shot_interval:04d}.npy') for i in range(n_shots)
		]
		observed_scatter_masked = [torch.tensor(np.load(f), device=self.device) for f in observed_scatter_files]

		for epoch in tqdm(range(n_epochs), desc="Epochs"):
			epoch_loss = 0
			for shot_index in tqdm(range(n_shots), desc="Shots", leave=False):
				current_source_position = shot_index * shot_interval
				source_locations = torch.zeros(1, 1, 2, dtype=torch.long, device=self.device)
				source_locations[0, 0, 0] = current_source_position

				out = scalar_born(
					self.vp, self.scatter, self.dx, self.dt,
					source_amplitudes=self.source_amplitudes,
					source_locations=source_locations,
					receiver_locations=self.receiver_locations,
					pml_freq=self.freq
				)

				observed_scatter_shot = observed_scatter_masked[shot_index]
				loss = self.loss_fn(out[-1], observed_scatter_shot)
				epoch_loss += loss.item()
				loss.backward()

			self.optimizer.step()
			self.optimizer.zero_grad()
			print(f'Epoch {epoch}, Loss: {epoch_loss}')

		scatter_numpy = self.scatter.detach().cpu().numpy()
		np.save(os.path.join(self.npy_folder, 'migration.npy'), scatter_numpy)

	def load_and_clip_data(self, clip_percent=99):
		migration_data = np.load(os.path.join(self.npy_folder, 'migration.npy'))
		migration_data -= np.mean(migration_data)
		migration_data /= np.max(np.abs(migration_data))
		# migration_data_clipped = migration_data[:, 130:] # Clip top image
		migration_data_clipped = migration_data[:, 95:] # Clip top image
		print("Shape of migration_data_clipped:", migration_data_clipped.shape)
		vmax_clip, vmin_clip = U.clip(migration_data_clipped, clip_percent)
		plt.figure()
		plt.imshow(migration_data_clipped.T, aspect='auto', cmap='gray', vmin=vmin_clip, vmax=vmax_clip)
		plt.xlabel('Distance (pixel)')
		plt.ylabel('Depth (m)')
		plt.title('Migration')
		plt.show()
