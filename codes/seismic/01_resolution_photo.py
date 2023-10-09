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
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

#-----------------------------------------------------------------------------------------#

if slice_index >= img_array.shape[1]:
    raise ValueError(f"slice_index {slice_index} is out of bounds for axis 1 with size {img_array.shape[1]}")
else:
    y = np.arange(img_array.shape[0])
    channels = ['Red', 'Green', 'Blue']
    colors = ['r', 'g', 'b']
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(img_array, aspect='auto')
    ax[0].set_title('Original Image')
    ax[0].set_xlabel('X')
    ax[0].set_ylabel('Y')
    ax[0].axvline(x=slice_index, color='r', linestyle='--')
    for i, (channel, color) in enumerate(zip(channels, colors)):
        trace = img_array[:, slice_index, i].astype(float)
        trace = trace - np.mean(trace)
        trace = trace / np.max(np.abs(trace))
        ax[1].plot(trace, y, label=channel, color=color)
    ax[1].invert_yaxis()  
    ax[1].set_title('Extracted Traces')
    ax[1].set_xlabel('Normalized Amplitude')
    ax[1].set_ylabel('Depth')
    ax[1].legend()  # Adding a legend
    plt.tight_layout()
    # plt.savefig(output_folder + "/photo_resolution.svg", format='svg', bbox_inches='tight', pad_inches=0, transparent=True)
    plt.show()

#-----------------------------------------------------------------------------------------#