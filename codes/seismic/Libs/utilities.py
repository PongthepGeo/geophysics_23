#-----------------------------------------------------------------------------------------#
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import signal
import cv2
from scipy.ndimage import gaussian_filter
import deepwave
from deepwave import scalar
import matplotlib.patches as patches
#-----------------------------------------------------------------------------------------#
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

def ricker(frequency, length=0.128, dt=0.004): 
	time = np.arange(-length/2, (length-dt)/2, dt)
	wiggle = (1.0 - 2.0*(np.pi**2)*(frequency**2)*(time**2)) * np.exp(-(np.pi**2)*(frequency**2)*(time**2))
	return wiggle

def clip(model, perc):
	(ROWs, COLs) = model.shape
	reshape2D_1D = model.reshape(ROWs*COLs)
	reshape2D_1D = np.sort(reshape2D_1D)
	if perc != 100:
		min_num = reshape2D_1D[ round(ROWs*COLs*(1-perc/100)) ]
		max_num = reshape2D_1D[ round((ROWs*COLs*perc)/100) ]
	elif perc == 100:
		min_num = min(model.flatten())
		max_num = max(model.flatten())
	if min_num > max_num:
		dummy = max_num
		max_num = min_num
		min_num = dummy
	return max_num, min_num 

def reflectivity(vp, frequency):
	wiggle = ricker(frequency)
	(ROWs, COLs) = vp.shape
	reflectivity = np.zeros_like(vp, dtype='float')
	conv = np.zeros_like(vp, dtype='float')
	rho = 2700
	for col in range (0, COLs):
		for row in range (0, ROWs-1):
			reflectivity[row, col] = (vp[row+1, col]*rho - vp[row, col]*rho) / (vp[row+1, col]*rho + vp[row, col]*rho)
		# flip polarity
		conv[:, col] = signal.convolve((reflectivity[:, col]*-1), wiggle, mode='same') / sum(wiggle)
	laplacian = cv2.Laplacian(conv, cv2.CV_64F)
	return laplacian

def photo2seismic(img, slice_index, output_folder):
	# NOTE Compute the seismic trace from a photo
	img = img / 255.0 if img.max() > 1.0 else img
	averaged_img = np.mean(img, axis=-1)
	old_min = np.min(averaged_img)
	old_max = np.max(averaged_img)
	new_min = 2; new_max = 5
	normalized_img = new_min + ((averaged_img - old_min) * (new_max - new_min)) / (old_max - old_min)
	smooth = gaussian_filter(normalized_img, sigma=(10, 10))
	laplacian = reflectivity(smooth, 30)
	max_num, min_num = clip(laplacian, 99)
	# NOTE Plotting the seismic trace
	if slice_index >= laplacian.shape[1]:
		raise ValueError(f"slice_index {slice_index} is out of bounds for axis 1 with size {laplacian.shape[1]}")
	else:
		y = np.arange(laplacian.shape[0])
		fig, ax = plt.subplots(1, 2, figsize=(12, 6))
		img_show = ax[0].imshow(laplacian, cmap='gray', vmin=min_num, vmax=max_num)
		ax[0].set_title('Processed Image')
		ax[0].set_xlabel('X')
		ax[0].set_ylabel('Y')
		plt.colorbar(img_show, ax=ax[0], label='Normalized Intensity')
		ax[0].axvline(x=slice_index, color='r', linestyle='--')
		trace = laplacian[:, slice_index].astype(float)
		trace = trace - np.mean(trace)
		trace = trace / np.max(np.abs(trace))
		ax[1].plot(trace, y, label='Trace', color='k')
		ax[1].invert_yaxis()  
		ax[1].set_title('Extracted Trace')
		ax[1].set_xlabel('Normalized Amplitude')
		ax[1].set_ylabel('Depth')
		ax[1].legend()  # Adding a legend
		plt.tight_layout()
		# plt.savefig(output_folder + "/seismic_resolution.svg", format='svg', bbox_inches='tight', pad_inches=0, transparent=True)
		# plt.show()
	return laplacian, trace

def DFT(x):
	N = len(x)
	n = np.arange(N)
	k = n.reshape((N, 1))
	e = np.exp(-2j * np.pi * k * n / N)
	X = np.dot(e, x)
	return X

def plot_spectra(X, sampling_rate): 
	# calculate the frequency
	N = len(X)
	n = np.arange(N)
	T = N/sampling_rate
	freq = n/T 
	half = int(len(freq)/2)
	freq = freq[:half]
	X = X[:half]
	fig = plt.figure(figsize=(12, 8))  
	plt.stem(freq, abs(X), 'b', markerfmt=" ", basefmt="-b")
	plt.xlim(0, 30)
	plt.xlabel('frequency (Hz)')
	plt.ylabel('amplitude')
	# plt.savefig('image_out/' + 'DFT_2' + '.svg', format='svg', bbox_inches='tight', transparent=True, pad_inches=0)
	plt.show()

def get_wavefield(vp, dx, dt, freq, nt, device, source_location):
	peak_time = 1.5 / freq # The time at which the Ricker wavelet reaches its peak
	return scalar(vp,            # Velocity model
				  dx,            # Spatial sampling interval
				  dt,            # Temporal sampling interval
				  source_amplitudes=(
				  deepwave.wavelets.ricker(freq, nt, dt, peak_time).reshape(1, 1, -1).to(device)), # Source wavelet
				  source_locations=source_location,  # Location of the source in the grid
				  accuracy=8,    # Accuracy of the finite difference stencil
				  pml_freq=freq) # Perfectly Matched Layer frequency to absorb boundary reflections

def plot_wave_propagation(vp, dx, dt, freq, time_steps, device, source_location, output_folder):
	plt.figure(figsize=(10, 10))  
	wavefields = [get_wavefield(vp, dx, dt, freq, nt, device, source_location) for nt in time_steps]
	# Extract source location from the tensor
	pml_thickness = 20
	source_y = (source_location[0, 0, 0] + pml_thickness).item()
	source_x = (source_location[0, 0, 1] + pml_thickness).item()
	for idx, (wavefield, nt) in enumerate(zip(wavefields, time_steps), 1):
		plt.subplot(2, 2, idx)
		wave_data = wavefield[0][0, :, :].cpu().numpy() # extract array from tensor and move to CPU
		max_num, min_num = clip(wave_data, 98)
		plt.imshow(wave_data, cmap='gray', vmin=min_num, vmax=max_num)
		plt.scatter(source_x, source_y, c='blue', s=50)  # Plot blue dot at source location
		plt.xlabel('X Distance (m)')
		plt.ylabel('Y Distance (m)')
		plt.title(f"Time Step: {nt} ms")
	plt.subplots_adjust(wspace=0.1, hspace=0.4)  
	# plt.savefig(output_folder + "/multiple.svg", format='svg', bbox_inches='tight', pad_inches=0, transparent=True)
	plt.show()

def box(vp, dx, dt, freq, time_steps, device, source_location, box_start_x, box_start_y,
		box_end_x, box_end_y, output_folder):
	plt.figure(figsize=(10, 10))
	wavefields = [get_wavefield(vp, dx, dt, freq, nt, device, source_location) for nt in time_steps]
	pml_thickness = 20
	source_y = (source_location[0, 0, 0] + pml_thickness).item()
	source_x = (source_location[0, 0, 1] + pml_thickness).item()
	box_start_x += pml_thickness; box_end_x += pml_thickness
	box_start_y += pml_thickness; box_end_y += pml_thickness
	for idx, (wavefield, nt) in enumerate(zip(wavefields, time_steps), 1):
		plt.subplot(2, 2, idx)
		wave_data = wavefield[0][0, :, :].cpu().numpy() # extract array from tensor and move to CPU
		max_num, min_num = clip(wave_data, 98)
		plt.imshow(wave_data, cmap='gray', vmin=min_num, vmax=max_num)
		plt.scatter(source_x, source_y, c='blue', s=50) 
		rect = patches.Rectangle((box_start_x, box_start_y),
								box_end_x - box_start_x,
								box_end_y - box_start_y,
								linewidth=1, edgecolor='r', facecolor='none')
		plt.gca().add_patch(rect)
		plt.xlabel('X Distance (m)')
		plt.ylabel('Y Distance (m)')
		plt.title(f"Time Step: {nt} ms")
	plt.subplots_adjust(wspace=0.1, hspace=0.4)
	# plt.savefig(output_folder + "/box.svg", format='svg', bbox_inches='tight', pad_inches=0, transparent=True)
	plt.show()

def wavefield(freq, nt, dt, peak_time, n_shots, n_sources_per_shot, device, vp, dx,
			  source_locations, receiver_locations):
	source_amplitudes = (deepwave.wavelets.ricker(freq, nt, dt, peak_time)
						 .repeat(n_shots, n_sources_per_shot, 1).to(device))
	outputs = scalar(vp, dx, dt,
					 source_amplitudes=source_amplitudes,
					 source_locations=source_locations,
					 receiver_locations=receiver_locations,
					 accuracy=8,
					 pml_width=[40, 40, 40, 40],
					 pml_freq=freq)
	wavefields, receiver_amplitudes = outputs[0], outputs[-1]  
	return wavefields, receiver_amplitudes

def plot_receivers(freq, dt, peak_time, n_shots, n_sources_per_shot, device, vp, dx, source_locations, 
				   receiver_locations, time_steps, limestone_start, output_folder):
	plt.figure()
	for i, nt in enumerate(time_steps):
		wave_propagation, receiver_data = wavefield(freq, nt, dt, peak_time, n_shots, n_sources_per_shot,
													device, vp, dx, source_locations, receiver_locations)
		wave_propagation = wave_propagation[0, :, :].cpu().numpy().T
		receiver_data = receiver_data[0].cpu().numpy().T
		# NOTE Wave Propagation
		plt.subplot(2, 2, 2*i + 1)
		y_max_wp = wave_propagation.shape[0] * dx * 0.001  # Convert y index to kilometers for wave propagation
		x_max_wp = wave_propagation.shape[1] * dx * 0.001  # Convert x index to kilometers for wave propagation
		max_wp, min_wp = clip(wave_propagation, 100)
		plt.imshow(wave_propagation, aspect='auto', cmap='gray', origin='upper',
				   extent=[0, x_max_wp, y_max_wp, 0], vmin=min_wp, vmax=max_wp)
		pml_thickness = 40
		limestone_depth_km = (limestone_start + pml_thickness) * dx * 0.001
		plt.axhline(limestone_depth_km, color='salmon', linestyle='--', linewidth=2)
		plt.title(f"Wave Propagation: {nt*0.001} s")
		plt.xlabel('Distance (km)')
		plt.ylabel('Depth (km)')
		# NOTE Receiver Data
		plt.subplot(2, 2, 2*i + 2)
		nt_seconds = nt * 0.001  # Convert nt to seconds
		max_rd, min_rd = clip(receiver_data, 99.5)
		plt.imshow(receiver_data, aspect='auto', cmap='gray', origin='upper',
				   extent=[0, x_max_wp, nt_seconds, 0], vmin=min_rd, vmax=max_rd)
		plt.xlabel('Receiver Position (km)')
		plt.ylabel('Time (sec)')
		plt.title('Receiver')
	plt.subplots_adjust(wspace=0.4, hspace=0.6)
	# plt.savefig(output_folder + "/receivers_2_layers.svg", format='svg', bbox_inches='tight', pad_inches=0, transparent=True)
	plt.show()