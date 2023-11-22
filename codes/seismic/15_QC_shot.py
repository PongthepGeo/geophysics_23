#-----------------------------------------------------------------------------------------#
import sys
sys.path.append('./Libs') 
import utilities as U
#-----------------------------------------------------------------------------------------#
import numpy as np
import matplotlib.pyplot as plt
import os
#-----------------------------------------------------------------------------------------#

receiver_data = np.load('npy_gold_folder/shot_pixel_0100.npy')
print("Shape of receiver_data:", receiver_data.shape)

#-----------------------------------------------------------------------------------------#

receiver_data = receiver_data[0, ...]
max_num, min_num = U.clip(receiver_data, 95)
plt.imshow(receiver_data.T, aspect='auto', cmap='gray', origin='upper', vmin=min_num, vmax=max_num)
plt.show()

#-----------------------------------------------------------------------------------------#

