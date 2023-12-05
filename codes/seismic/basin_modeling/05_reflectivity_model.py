#-----------------------------------------------------------------------------------------#
from Libs import utilities as U
#-----------------------------------------------------------------------------------------#
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
#-----------------------------------------------------------------------------------------#

image_path = "models/basin_01.png"
slice_index = 500
output_folder = "image_out"

#-----------------------------------------------------------------------------------------#

img = Image.open(image_path)
img_array = np.array(img)
U.photo2seismic(img_array, slice_index, output_folder)

#-----------------------------------------------------------------------------------------#