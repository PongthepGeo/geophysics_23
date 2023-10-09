#-----------------------------------------------------------------------------------------#
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import cv2
from scipy.ndimage import gaussian_filter
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
        plt.savefig(output_folder + "/seismic_resolution.svg", format='svg', bbox_inches='tight', pad_inches=0, transparent=True)
        plt.show()