#-----------------------------------------------------------------------------------------#
from Libs import utilities as U
#-----------------------------------------------------------------------------------------#
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math
#-----------------------------------------------------------------------------------------#

image_path = "data/resolution_photo.png"
slice_index = 770 
output_folder = "image_out"

#-----------------------------------------------------------------------------------------#

img = Image.open(image_path)
img_array = np.array(img)
seismic_like_image, _ = U.photo2seismic(img_array, slice_index, output_folder)

#-----------------------------------------------------------------------------------------#

def DFT(x, sampling_rate):
	N = len(x)
	n = np.arange(N)
	k = n.reshape((N, 1))
	e = np.exp(-2j * np.pi * k * n / N)
	X = np.dot(e, x)
	N = len(X)
	n = np.arange(N)
	T = N/sampling_rate
	freq = n/T 
	half = int(len(freq)/2)
	freq = freq[:half]
	X = X[:half]
	return abs(X)

def avg_index_top_n(arr, n=5):
	sorted_indices = np.argsort(arr)
	top_n_indices = sorted_indices[-n:]
	return np.mean(top_n_indices)

#-----------------------------------------------------------------------------------------#

ROWs, COLs = seismic_like_image.shape
sampling_rate = COLs

for i in range (0, COLs):
	X = DFT(seismic_like_image[:, i], sampling_rate)
	avg_index = avg_index_top_n(X, 5)
	print(f"Average frequency for top 5 values: {avg_index} Hz of trace: {i}")

#-----------------------------------------------------------------------------------------#