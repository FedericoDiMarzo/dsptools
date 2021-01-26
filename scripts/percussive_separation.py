from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import sounddevice as sd
from dsp.processing import median_separation, normalize

# %% importing audio file
path = Path().joinpath('media', 'audio', 'mixdowns', 'disco0.wav')
fs, audio = wavfile.read(path)
audio = np.sum(audio, axis=1)  # mono sum
audio = audio / np.max(audio)  # normalization
t_begin = 5
t_end = 10
sample_begin = t_begin * fs
sample_end = t_end * fs
audio = audio[sample_begin:sample_end]  # resizing
t = np.linspace(0, len(audio)/fs, len(audio))

plt.subplot(311)
plt.plot(t, audio)
plt.title('original audio')

#%% percussive mask extraction
percussive, harmonic = median_separation(audio, fs)
percussive = normalize(percussive)
harmonic = normalize(harmonic)

plt.subplot(312)
plt.plot(t, percussive)
plt.title('percussive')

plt.subplot(313)
plt.plot(t, harmonic)
plt.title('harmonic')
plt.show()

#%% play sounds
sd.play(audio, fs)
sd.wait()
sd.play(percussive, fs)
sd.wait()
sd.play(harmonic, fs)
sd.wait()
