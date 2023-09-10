#-----------------------------------------------------------------------------------------#
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image
from sklearn.cluster import KMeans
from scipy.ndimage import gaussian_filter
#-----------------------------------------------------------------------------------------#

img_path = "image_out/gravity_model_02.png"
# img_path = "image_out/ex_gravity_model.png"
output_dir = "data_out"

#-----------------------------------------------------------------------------------------#

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

#-----------------------------------------------------------------------------------------#

image = Image.open(img_path)
data = np.array(image)
data = data[:, :, :3]
data_subsection = data[700:]
# data_subsection = data
smoothed_data = gaussian_filter(data_subsection, sigma=(4, 4, 0))
pixels = smoothed_data.reshape(-1, 3)
kmeans = KMeans(n_clusters=5, n_init='auto', max_iter=10000, random_state=42)
kmeans.fit(pixels)
labels = kmeans.predict(pixels)

#-----------------------------------------------------------------------------------------#

labels_image = labels.reshape(smoothed_data.shape[0], smoothed_data.shape[1])
colors = [
    [66/255, 73/255, 73/255],    # mud (gray) 2100
    [39/255, 55/255, 70/255],    # shale (dark gray) 2700
    [244/255, 208/255, 63/255],  # sand (yellow) 3120
    [255/255, 127/255, 80/255],  # gold (orange) 19300
    [125/255, 102/255, 8/255]    # shaly-sand (dark yellow) 2800
]
cmap = ListedColormap(colors)
fig, ax = plt.subplots(figsize=(labels_image.shape[1]/80, labels_image.shape[0]/80))  # 80 dpi is default for most monitors
img_obj = ax.imshow(labels_image, cmap=cmap)
cbar = plt.colorbar(img_obj, ticks=range(5), orientation='vertical', ax=ax)
cbar.set_label('Labels')
label_names = ['Mud', 'Shale', 'Sand', 'Gold', 'Shaly-Sand']
cbar.set_ticklabels(label_names)
# plt.savefig("data_out/gravity_model.svg", format='svg', bbox_inches='tight', pad_inches=0, transparent=True, dpi=80)
plt.show()

#-----------------------------------------------------------------------------------------#

density_map = {0: 2100, 1: 2700, 2: 3120, 3: 19300, 4: 2800}
densities_image = np.vectorize(density_map.get)(labels_image)
np.save(os.path.join(output_dir, "densities_image.npy"), densities_image)
# np.save(os.path.join(output_dir, "exercise.npy"), densities_image)

#-----------------------------------------------------------------------------------------#