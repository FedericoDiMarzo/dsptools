from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
import sounddevice as sd
from librosa.decompose import hpss

"""
This example shows the usage of the HPSS algorithm for
separating percussive and harmonic parts from a source
"""

# %% importing audio file
path = Path('..').joinpath('media', 'audio', 'mixdowns', 'disco0.wav')
fs, audio = wavfile.read(path)
audio = np.sum(audio, axis=1)  # mono sum
audio = audio / np.max(audio)  # normalization
t_begin = 5
t_end = 10
sample_begin = t_begin * fs
sample_end = t_end * fs
audio = audio[sample_begin:sample_end]  # resizing
t = np.linspace(0, len(audio) / fs, len(audio))

# %% STFT and HPSS
_, _, X = signal.stft(audio, return_onesided=True)
H, P = hpss(X)
_, p = signal.istft(P, input_onesided=True)
_, h = signal.istft(H, input_onesided=True)
p = p[0:len(audio)]
h = h[0:len(audio)]

# %% plotting
plt.subplot(211)
plt.plot(p)
plt.subplot(212)
plt.plot(h)
plt.show()

# %% numerical tests
print('difference between original and sum: {}'
      .format(np.mean(audio - p - h)))

# %% sound
sd.play(audio, fs)
sd.wait()
sd.play(p, fs)
sd.wait()
sd.play(h, fs)
sd.wait()


