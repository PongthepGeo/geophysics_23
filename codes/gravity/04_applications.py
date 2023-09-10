import os
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Access the gravity_dataset main folder
base_dir = "gravity_datasets"

# Step 2: Access Thailand_SHP and read shape files
thailand_shp_path = os.path.join(base_dir, "Thailand_SHP")
thailand_boundary = gpd.read_file(thailand_shp_path)

# Step 3: Access GRACE_Thailand and read TIFF
grace_thailand_path = os.path.join(base_dir, "GRACE_Thailand")
tiff_files = [f for f in os.listdir(grace_thailand_path) if f.endswith('.TIFF') or f.endswith('.tif')]
# Ensure that there's at least one TIFF file
if not tiff_files:
    raise ValueError("No TIFF files found in GRACE_Thailand directory.")
tiff_file_path = os.path.join(grace_thailand_path, tiff_files[0])
with rasterio.open(tiff_file_path) as src:
    img_extent = [src.bounds[0], src.bounds[2], src.bounds[1], src.bounds[3]]
    img = src.read(1)

# Check and reproject if necessary
if thailand_boundary.crs != src.crs:
    thailand_boundary = thailand_boundary.to_crs(src.crs)
    
# Step 4: Plot TIFF with Thailand boundary
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(img, cmap='gray', extent=img_extent)
thailand_boundary.boundary.plot(ax=ax, color='red')

# Define the number of desired tick marks for both x and y
num_ticks = 10

# Calculate the interval based on extent and desired number of tick marks
x_interval = (img_extent[1] - img_extent[0]) / num_ticks
y_interval = (img_extent[3] - img_extent[2]) / num_ticks

x_ticks = np.arange(img_extent[0], img_extent[1] + x_interval, x_interval)
y_ticks = np.arange(img_extent[2], img_extent[3] + y_interval, y_interval)

ax.set_xticks(x_ticks)
ax.set_yticks(y_ticks)

ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.grid(which='both', linestyle='--', linewidth=0.5, color='blue', alpha=0.5)

ax.set_title('Thailand GRACE Data with Boundary Overlay')
plt.show()
