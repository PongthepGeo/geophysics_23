#-----------------------------------------------------------------------------------------#
import numpy as np
import matplotlib.pyplot as plt
#-----------------------------------------------------------------------------------------#

time = np.linspace(0, np.pi/32, 1000) 

#-----------------------------------------------------------------------------------------#

amplitude = 2; frequency = 25; phase = 90 
signal_1 = amplitude*np.sin(2*np.pi*frequency*time + np.deg2rad(phase))  

#-----------------------------------------------------------------------------------------#

amplitude = 1; frequency = 30; phase = 10 
signal_2 = amplitude*np.sin(2*np.pi*frequency*time + np.deg2rad(phase))  

#-----------------------------------------------------------------------------------------#

amplitude = 2; frequency = 50; phase = 45 
signal_3 = amplitude*np.sin(2*np.pi*frequency*time + np.deg2rad(phase))  

#-----------------------------------------------------------------------------------------#

amplitude = 2; frequency = 45; phase = 0 
signal_4 = amplitude*np.sin(2*np.pi*frequency*time + np.deg2rad(phase))  

#-----------------------------------------------------------------------------------------#

sum_signal = signal_1 + signal_2 + signal_3 + signal_4

fig = plt.figure(figsize=(10, 5))
plt.plot(time, signal_1, color='red', linestyle='--', linewidth=2)
plt.plot(time, signal_2, color='blue', linestyle='--', linewidth=2)
plt.plot(time, signal_3, color='green', linestyle='--', linewidth=2)
plt.plot(time, signal_4, color='black', linestyle='--', linewidth=2)
plt.plot(time, sum_signal, color='cyan', linestyle='-', linewidth=6)
plt.xlabel('Time (s)', fontsize=18)
plt.ylabel('Amplitude', fontsize=18)
plt.title('Sine Wave', fontsize=18, fontweight='bold')
# plt.savefig('image_out/' + 'super' + '.svg', format='svg', bbox_inches='tight', transparent=True, pad_inches=0)
plt.show()

#-----------------------------------------------------------------------------------------#
