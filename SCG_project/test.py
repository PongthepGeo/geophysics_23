import numpy as np
import matplotlib.pyplot as plt
from readgssi import readgssi

def get_data(infile):
    data_tuple = readgssi.readgssi(infile=infile, zero=[233], plotting=False, figsize=5)
    radar_dict = data_tuple[1]
    radar_data = radar_dict[0]
    if not isinstance(radar_data, np.ndarray):
        radar_data = np.array(radar_data)
    return radar_data

# Get data for each file
data_1_array = get_data('soil_data/Sep_05/TT001.DZT')
data_2_array = get_data('soil_data/Sep_05/TT002.DZT')
data_3_array = get_data('soil_data/Sep_05/TT003.DZT')
data_4_array = get_data('soil_data/Sep_05/TT004.DZT')

# Plotting
plt.figure(figsize=(10, 6))

# Create an array for Y axis values. It will be the same for all data arrays.
y = np.arange(data_1_array.shape[0])

# Since you want y-axis to start from min at the top, you can reverse the y-axis using:
plt.gca().invert_yaxis()

# Plot each data array with a unique color
plt.plot(data_1_array, y, label='Data 1', color='blue')
plt.plot(data_2_array, y, label='Data 2', color='red')
plt.plot(data_3_array, y, label='Data 3', color='green')
plt.plot(data_4_array, y, label='Data 4', color='yellow')

plt.legend()
plt.title('Radar Data Visualization')
plt.xlabel('Radar Amplitude')
plt.ylabel('Depth/Time')
plt.grid(True)

plt.show()