from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import sounddevice as sd
from dsp.processing import denoise

ylim = [-150, 80]

# %% importing audio file
path = Path().joinpath('media', 'audio', 'female_voice', 'all night mid.wav')
fs, x = wavfile.read(path)
x = x / np.max(x) * 0.75  # normalization
t_max = 4
sample_max = t_max * fs
x = x[0:sample_max]  # resizing

# %% applying noise
noise = np.random.normal(0, 0.1, len(x))
x_noisy = x + noise

# %% noisy periodogram
f, Pxn = signal.periodogram(x_noisy, nfft=1024)
plt.subplot(211)
plt.title('denoising')
plt.plot(f * fs, 10 * np.log(Pxn))
plt.xscale('log')
plt.ylim(ylim)
plt.grid()

# %% filtering the noise
y = denoise(x, noise)
f, Py = signal.periodogram(y, nfft=1024)
plt.subplot(212)
plt.plot(f * fs, 10 * np.log(Py))
plt.xscale('log')
plt.ylim(ylim)
plt.grid()

plt.show()

# %% play sounds
sd.play(x_noisy, fs)
sd.wait()
sd.play(y, fs)
sd.wait()
