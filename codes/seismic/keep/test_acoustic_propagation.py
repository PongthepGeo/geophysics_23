#-----------------------------------------------------------------------------------------#
import torch
import matplotlib.pyplot as plt
import deepwave
from deepwave import scalar
import numpy as np
import warnings
#-----------------------------------------------------------------------------------------#
warnings.filterwarnings("ignore", category=UserWarning)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
#-----------------------------------------------------------------------------------------#

ny = 500; nx = 500
vp = 2500 * torch.ones(ny, nx)

# NOTE in case for QC input velocity
# v = vp.cpu().numpy()
# plt.imshow(np.rot90(v, 3), cmap='gray', vmin=2000, vmax=5000)  
# plt.show()

# n_shots = 115
# n_sources_per_shot = 1
# d_source = 20  # 20 * 4m = 80m
# first_source = 10  # 10 * 4m = 40m
# source_depth = 2  # 2 * 4m = 8m

n_shots = 1
n_sources_per_shot = 1
d_source = 20  # 20 * 4m = 80m
first_source = nx // 2  # 10 * 4m = 40m
source_depth = ny // 2  # 2 * 4m = 8m

freq = 25
nt = 750
dt = 0.004
peak_time = 1.5 / freq
dx = 4.0

# source_locations
source_locations = torch.zeros(n_shots, n_sources_per_shot, 2, dtype=torch.long, device=device)
source_locations[..., 1] = source_depth
# source_locations[:, 0, 0] = (torch.arange(n_shots) * d_source +
                            #  first_source)

# source_amplitudes
source_amplitudes = (deepwave.wavelets.ricker(freq, nt, dt, peak_time)
                    .repeat(n_shots, n_sources_per_shot, 1)
                    .to(device))

out = scalar(vp, dx, dt, source_amplitudes=source_amplitudes,
             source_locations=source_locations,
             accuracy=8,
             pml_freq=freq)

_, ax = plt.subplots(4, figsize=(3.5, 5), sharey=True,
                     sharex=True)
ax[0].imshow(out[0][0, :, :], cmap='gray')
plt.show()

# receiver_amplitudes = out[-1]
# vmin, vmax = torch.quantile(receiver_amplitudes[0],
#                             torch.tensor([0.05, 0.95]).to(device))
# _, ax = plt.subplots(1, 2, figsize=(10.5, 7), sharey=True)
# ax[0].imshow(receiver_amplitudes[57].cpu().T, aspect='auto',
#              cmap='gray', vmin=vmin, vmax=vmax)
# ax[1].imshow(receiver_amplitudes[:, 192].cpu().T, aspect='auto',
#              cmap='gray', vmin=vmin, vmax=vmax)
# ax[0].set_xlabel("Channel")
# ax[0].set_ylabel("Time Sample")
# ax[1].set_xlabel("Shot")
# plt.tight_layout()
# plt.show()