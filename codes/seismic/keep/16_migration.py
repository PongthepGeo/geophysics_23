#-----------------------------------------------------------------------------------------#
import sys
sys.path.append('./Libs') 
import classes as C
import utilities as U
#-----------------------------------------------------------------------------------------#
import numpy as np
from PIL import Image
import torch
import deepwave
from deepwave import scalar
import matplotlib.pyplot as plt
import os
from deepwave import scalar_born
from tqdm import tqdm
#-----------------------------------------------------------------------------------------#

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

#-----------------------------------------------------------------------------------------#

# NOTE Predefine velocity model parameters
image_path = 'data/modeling/tilting_03.png'
minimum_velocity = 1500
maximum_velocity = 4500
smooth = 20
# NOTE Predefine source and receiver parameters
output_folder = "image_out"
freq = 25               # Frequency of the source in Hz 
dx = 4.0                # Spatial sampling interval in meters 
dt = 0.004              # Temporal sampling interval in seconds
peak_time = 1.5 / freq
nt = 700
output_migration = 'migrated_image'
shot_interval = 10      # Every 10 pixel will allocate 1 shot
npy_folder = 'npy_folder'  # Load shot data from this folder

#-----------------------------------------------------------------------------------------#

os.makedirs(output_migration, exist_ok=True)

#-----------------------------------------------------------------------------------------#

# NOTE Image to velocity model conversion
img = Image.open(image_path)
processor = C.Image2Velocity(img, smooth)
vp_array = processor.plot_velocity(minimum_velocity, maximum_velocity, output_folder)
nx = vp_array.shape[1]

#-----------------------------------------------------------------------------------------#

vp = torch.tensor(vp_array, dtype=torch.float64).to(device)
vp = torch.transpose(vp, 0, 1)  # Transpose the model
vp = vp.to(device)

#-----------------------------------------------------------------------------------------#

# Initialize source and receiver parameters
n_shots = nx // shot_interval  # Number of shots based on model width and interval of 100 pixels
n_sources_per_shot = 1
source_depth = 2
source_locations = torch.zeros(1, n_sources_per_shot, 2, dtype=torch.long, device=device)  # Note the '1' for a single shot
source_locations[..., 1] = source_depth

#-----------------------------------------------------------------------------------------#

d_receiver = 3  # Receiver interval (grid points)
n_receivers_per_shot = nx // d_receiver
receiver_depth = 0
receiver_locations = torch.zeros(1, n_receivers_per_shot, 2, dtype=torch.long, device=device)
receiver_locations[..., 1] = receiver_depth
receiver_locations[0, :, 0] = torch.arange(0, nx, d_receiver).long()[:n_receivers_per_shot]

#-----------------------------------------------------------------------------------------#

source_amplitudes = deepwave.wavelets.ricker(freq, nt, dt, peak_time).reshape(1, 1, -1).to(dtype=torch.float64, device=device)

#-----------------------------------------------------------------------------------------#

# Create scattering amplitude that we will invert for
scatter = torch.zeros_like(vp)
scatter.requires_grad_()

# Setup optimiser to perform inversion
# optimiser = torch.optim.SGD([scatter], lr=1e9)
# optimiser = torch.optim.SGD([scatter], lr=1e4)
optimiser = torch.optim.Adam([scatter], lr=1e-4)
# optimiser = torch.optim.RMSprop([scatter], lr=1e-3)
# optimiser = torch.optim.Adagrad([scatter], lr=1e-2)
# optimiser = torch.optim.AdamW([scatter], lr=1e-3)


# loss_fn = torch.nn.MSELoss()
loss_fn = torch.nn.CrossEntropyLoss()
# loss_fn = torch.nn.BCEWithLogitsLoss()
# loss_fn = torch.nn.NLLLoss()
# loss_fn = torch.nn.L1Loss()


# Load observed scatter data from npy files
observed_scatter_files = [os.path.join(npy_folder, f'shot_pixel_{i * shot_interval:04d}.npy') for i in range(n_shots)]
observed_scatter_masked = [torch.tensor(np.load(f), device=device) for f in observed_scatter_files]

n_epochs = 1

for epoch in tqdm(range(n_epochs), desc="Epochs"):
    epoch_loss = 0
    for shot_index in tqdm(range(n_shots), desc="Shots", leave=False):
        current_source_position = shot_index * shot_interval
        source_locations[0, 0, 0] = current_source_position

        # Run scalar_born for the current shot
        out = scalar_born(
            vp, scatter, dx, dt,
            source_amplitudes=source_amplitudes,
            source_locations=source_locations,
            receiver_locations=receiver_locations,
            pml_freq=freq
        )

        # Load the observed scatter for the current shot
        observed_scatter_shot = observed_scatter_masked[shot_index]

        # Compute loss for the current shot
        loss = loss_fn(out[-1], observed_scatter_shot)
        epoch_loss += loss.item()
        loss.backward()

    optimiser.step()
    optimiser.zero_grad()
    print(f'Epoch {epoch}, Loss: {epoch_loss}')

# Convert scatter to numpy array and save
scatter_numpy = scatter.detach().cpu().numpy()
np.save(os.path.join(npy_folder, 'migration.npy'), scatter_numpy)

# Load migration.npy
migration_data = np.load(os.path.join(npy_folder, 'migration.npy'))
print("Shape of migration_data:", migration_data.shape)

# Remove the first 50 rows
# migration_data_clipped = migration_data[600:, :]
# migration_data_clipped = migration_data[:, 200:]
migration_data_clipped = migration_data[:, :]
print("Shape of migration_data_clipped:", migration_data_clipped.shape)
vmax_clip, vmin_clip = U.clip(migration_data_clipped, 99)  # Example with 95% clip

# Plot the clipped data
plt.figure(figsize=(10.5, 3.5))
plt.imshow(migration_data_clipped.T, aspect='auto', cmap='gray', vmin=vmin_clip, vmax=vmax_clip)
plt.show()