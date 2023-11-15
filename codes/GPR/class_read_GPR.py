#-----------------------------------------------------------------------------------------#
import numpy as np
import matplotlib.pyplot as plt
import segyio
import matplotlib
import os
from readgssi import readgssi
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

class ClipMixin:
    def _clip(self, data, perc):
        rows, cols = data.shape
        flat_data = data.reshape(rows * cols)
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

class SEGYDataPlotter(ClipMixin):
    def __init__(self, segy_file):
        self.segy_file = segy_file
        self.default_title = os.path.splitext(os.path.basename(segy_file))[0].replace('_', ' ')
        self.data, self.twt, self.n_traces = self._read_segy_file()

    def _read_segy_file(self):
        with segyio.open(self.segy_file, "r", ignore_geometry=True) as segyfile:
            n_traces = segyfile.tracecount
            twt = segyfile.samples
            data = segyfile.trace.raw[:]
        return data, twt, n_traces

    def plot(self, clip_percent=90):
        plt.figure(figsize=(10, 6))
        max_num, min_num = self._clip(self.data, clip_percent)
        plt.imshow(self.data.T, extent=[0, self.n_traces, self.twt[-1], self.twt[0]],
                   aspect='auto', cmap='Greys', vmin=min_num, vmax=max_num)
        plt.colorbar(label='Amplitude')
        plt.xlabel('Trace number')
        plt.ylabel('Two-way travel time (ms)')
        plt.title(self.default_title)  # Use the default title
        plt.show()

class GPRDataPlotter(ClipMixin):
    def __init__(self, gpr_file):
        self.gpr_file = gpr_file
        self.data = self._read_gpr_file()

    def _read_gpr_file(self):
        data_tuple = readgssi.readgssi(infile=self.gpr_file, plotting=False)
        radar_dict = data_tuple[1]
        radar_data = radar_dict[0]
        if not isinstance(radar_data, np.ndarray):
            radar_data = np.array(radar_data)
        return radar_data

    def plot(self, clip_percent=90):
        plt.figure(figsize=(15, 10))
        max_num, min_num = self._clip(self.data, clip_percent)  
        plt.imshow(self.data, cmap='Greys', vmin=min_num, vmax=max_num)
        plt.colorbar(label='Amplitude')
        plt.xlabel('Trace number')
        plt.ylabel('Two-way travel time (ms)')
        plt.title('GPR at Accounting Department')  
        plt.show()