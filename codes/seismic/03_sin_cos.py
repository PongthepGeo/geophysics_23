#-----------------------------------------------------------------------------------------#
import numpy as np
import matplotlib.pyplot as plt
#-----------------------------------------------------------------------------------------#

amplitude = 2  # no unit
frequency = 25 # Hz
time = np.linspace(0, np.pi/32, 1000) # second
phase = 90 # degrees, equivalent to np.pi/2 radians

#-----------------------------------------------------------------------------------------#

# NOTE np.sin require input in radians
y = amplitude*np.sin(2*np.pi*frequency*time + np.deg2rad(phase))  # converting phase to radians

#-----------------------------------------------------------------------------------------#

fig = plt.figure(figsize=(10, 5))
plt.plot(time, y, color='red', linestyle='-', linewidth=4)
plt.xlabel('Time (s)', fontsize=18)
plt.ylabel('Amplitude', fontsize=18)
plt.title('Sine Wave', fontsize=18, fontweight='bold')
# plt.savefig('image_out/' + 'sine' + '.svg', format='svg', bbox_inches='tight', transparent=True, pad_inches=0)
plt.show()

#-----------------------------------------------------------------------------------------#