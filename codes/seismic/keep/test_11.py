import numpy as np
import matplotlib.pyplot as plt

trace = np.load('data/trace.npy')

plt.plot(trace)
plt.show()

def DFT(x):
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(-2j * np.pi * k * n / N)
    X = np.dot(e, x)
    return X

def plot_spectra(X, sampling_rate): 
	# calculate the frequency
	N = len(X)
	n = np.arange(N)
	T = N/sampling_rate
	freq = n/T 
	half = int(len(freq)/2)
	freq = freq[:half]
	X = X[:half]
	# print(X.shape)

	fig = plt.figure(figsize=(12, 8))  
	plt.stem(freq, abs(X), 'b', markerfmt=" ", basefmt="-b")
	plt.xlim(0, 30)
	plt.xlabel('frequency (Hz)')
	plt.ylabel('amplitude')
	# plt.savefig('image_out/' + 'DFT_2' + '.svg', format='svg', bbox_inches='tight', transparent=True, pad_inches=0)
	plt.show()

# TODO DFT
sampling_rate = 1000
# X = DFT(new_y)
X = DFT(trace)
plot_spectra(X, sampling_rate)