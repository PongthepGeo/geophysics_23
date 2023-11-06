#-----------------------------------------------------------------------------------------#
import numpy as np
import matplotlib.pyplot as plt
import segyio
import matplotlib
import os
#-----------------------------------------------------------------------------------------#
params = {
	'savefig.dpi': 300,  
	'figure.dpi' : 300,
	'axes.labelsize':12,  
	'axes.titlesize':12,
	'axes.titleweight': 'bold',
	'legend.fontsize': 10,
	'xtick.labelsize':10,
	'ytick.labelsize':10,
	'font.family': 'serif',
	'font.serif': 'Times New Roman'
}
matplotlib.rcParams.update(params)
#-----------------------------------------------------------------------------------------#

class SEGYDataPlotter:
    def __init__(self, segy_file):
        self.segy_file = segy_file
        # Set the default title by extracting the filename without the extension and replacing underscores with spaces.
        self.default_title = os.path.splitext(os.path.basename(segy_file))[0].replace('_', ' ')
        self.data, self.twt, self.n_traces = self._read_segy_file()

    def _read_segy_file(self):
        with segyio.open(self.segy_file, "r", ignore_geometry=True) as segyfile:
            n_traces = segyfile.tracecount
            twt = segyfile.samples
            data = segyfile.trace.raw[:]
        return data, twt, n_traces

    def _clip(self, perc):
        (rows, cols) = self.data.shape
        flat_data = self.data.reshape(rows * cols)
        sorted_data = np.sort(flat_data)
        if perc != 100:
            min_num = sorted_data[int(rows * cols * (1 - perc / 100))]
            max_num = sorted_data[int((rows * cols * perc) / 100)]
        else:
            min_num = np.min(flat_data)
            max_num = np.max(flat_data)
        if min_num > max_num:
            min_num, max_num = max_num, min_num  # Swap if out of order
        return max_num, min_num 

    def plot(self, clip_percent=90):
        plt.figure(figsize=(10, 6))
        max_num, min_num = self._clip(clip_percent)
        plt.imshow(self.data.T, extent=[0, self.n_traces, self.twt[-1], self.twt[0]],
                   aspect='auto', cmap='Greys', vmin=min_num, vmax=max_num)
        plt.colorbar(label='Amplitude')
        plt.xlabel('Trace number')
        plt.ylabel('Two-way travel time (ms)')
        plt.title(self.default_title)  # Use the default title
        plt.show()