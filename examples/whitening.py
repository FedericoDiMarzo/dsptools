from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
from dsp.processing import whiten

"""
In this example the periodogram of a signal is plotted before
and after the whitening process, showing how the spectrum flattens.
"""

ylim = [-150, 80]

# %% importing audio file
path = Path().joinpath('media', 'audio', 'mixdowns', 'PercPlusHarm.wav')
fs, x = wavfile.read(path)
x = np.sum(x, axis=1)  # mono sum
x = x / np.max(x)  # normalization
t_max = 4
sample_max = t_max * fs
x = x[0:sample_max]  # resizing

# %% plotting the PSD of x
f, Px = signal.periodogram(x, nfft=1024)

plt.subplot(211)
plt.title('whitening')
plt.plot(f * fs, 10 * np.log(Px))
plt.xscale('log')
plt.ylim(ylim)
plt.grid()

# %% whitening process
y = whiten(x)
f, Py = signal.periodogram(y, nfft=1024)

plt.subplot(212)
plt.plot(f * fs, 10 * np.log(Py))
plt.xscale('log')
plt.ylim(ylim)
plt.grid()
plt.show()

# %% play sounds
# sd.play(x, fs)
# sd.wait()
# sd.play(y, fs)
# sd.wait()
