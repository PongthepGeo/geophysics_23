#-----------------------------------------------------------------------------------------#
import numpy as np
import matplotlib.pyplot as plt
import math
#-----------------------------------------------------------------------------------------#

#-----------------------------------------------------------------------------------------#
from Libs import utilities as U
#-----------------------------------------------------------------------------------------#
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
#-----------------------------------------------------------------------------------------#

image_path = "data/resolution_photo.png"
slice_index = 770 
output_folder = "image_out"

#-----------------------------------------------------------------------------------------#

img = Image.open(image_path)
img_array = np.array(img)
trace = U.photo2seismic(img_array, slice_index, output_folder)
# print(trace)
np.save('data/trace.npy', trace)

#-----------------------------------------------------------------------------------------#

# # TODO construct complex waveforms from harmonic functions
# sampling_rate = 1000
# time = np.linspace(0, 1, sampling_rate) # 1000 milliseconds
# amplitude = [1.5, 2., 3.5] 
# frequency = [5, 2.5, 4.0] 
# degree = [0, 45, 90]
# color = ['r--', 'g--', 'b--']
# label = ['5 Hz', '2.5 Hz', '4.5 Hz']
# new_y = 0.

# # fig = plt.figure(figsize=(12, 8))  
# for index, i in enumerate(amplitude):
# 	print(index, i)
# 	angular_frequency = 2*np.pi*frequency[index] # radian per second
# 	phase_angle = math.radians(degree[index]) # radian (input degree)
# 	y = i * np.cos(angular_frequency*time + phase_angle)
# 	# plt.plot(y, color[index], linewidth=1.0, label=label[index])
# 	new_y += y
# plt.plot(new_y, 'y-', linewidth=3.0, label='summation')
# plt.xlabel('time (s)')
# plt.ylabel('amplitude')
# plt.legend(loc='upper right')
# # plt.savefig('image_out/' + 'DFT_1' + '.svg', format='svg', bbox_inches='tight', transparent=True, pad_inches=0)
# plt.show()

# def DFT(x):
#     N = len(x)
#     n = np.arange(N)
#     k = n.reshape((N, 1))
#     e = np.exp(-2j * np.pi * k * n / N)
#     X = np.dot(e, x)
#     return X

# def plot_spectra(X, sampling_rate): 
# 	# calculate the frequency
# 	N = len(X)
# 	n = np.arange(N)
# 	T = N/sampling_rate
# 	freq = n/T 
# 	half = int(len(freq)/2)
# 	freq = freq[:half]
# 	X = X[:half]
# 	# print(X.shape)

# 	fig = plt.figure(figsize=(12, 8))  
# 	plt.stem(freq, abs(X), 'b', markerfmt=" ", basefmt="-b")
# 	plt.xlim(0, 30)
# 	plt.xlabel('frequency (Hz)')
# 	plt.ylabel('amplitude')
# 	# plt.savefig('image_out/' + 'DFT_2' + '.svg', format='svg', bbox_inches='tight', transparent=True, pad_inches=0)
# 	plt.show()

# # TODO DFT
# # X = DFT(new_y)
# X = DFT(trace)
# plot_spectra(X, sampling_rate)