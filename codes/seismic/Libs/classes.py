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