#-----------------------------------------------------------------------------------------#
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
#-----------------------------------------------------------------------------------------#
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
#-----------------------------------------------------------------------------------------#

img = Image.open('data/resolution_photo.png')
ny, nx = 2301, 751

#-----------------------------------------------------------------------------------------#

img = img.resize((ny, nx), Image.LANCZOS)  
img = img.convert('L')
numpy_array = np.array(img)
new_min, new_max = 2000, 5000
numpy_array = (numpy_array / 255.0) * (new_max - new_min) + new_min
numpy_array = np.rot90(numpy_array)

#-----------------------------------------------------------------------------------------#

tensor = torch.from_numpy(numpy_array.copy()).float()
tensor = tensor.to(device=device)
# tensor = tensor.cpu().numpy() # QC before saving
# NOTE save bin
tensor.cpu().numpy().tofile('data/modeling/velocity.bin')
# NOTE load bin
v = torch.from_file('data/modeling/velocity.bin', size=ny*nx).reshape(ny, nx).to(device)

#-----------------------------------------------------------------------------------------#

v = v.cpu().numpy()
plt.imshow(np.rot90(v, 3), cmap='gray', vmin=2000, vmax=5000)  
plt.show()

#-----------------------------------------------------------------------------------------#