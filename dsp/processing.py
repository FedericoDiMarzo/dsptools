import numpy as np
from scipy import fft, signal, ndimage


def denoise(x, noise, segment_len=256, fft_resolution=1024):
    """
    Applies a Weiner filter to the input to reduce the noise
    :param x: input signal
    :param noise: sample of noise signal
    :param segment_len: analysis and synthesis window lengths
    :param fft_resolution: STFT frequency resolution
    :return: denoised signal
    """
    _, _, X = signal.stft(x, nperseg=segment_len,
                          nfft=fft_resolution,
                          return_onesided=True)
    _, Pv = signal.periodogram(x, nfft=fft_resolution)
    Pv = Pv.reshape(len(Pv), -1)
    Px = np.power(np.abs(X), 2)  # instantaneous power spectrum
    H = 1 - Pv / Px  # Weiner filter
    Y = X * H
    _, y = signal.istft(Y, nperseg=segment_len,
                        nfft=fft_resolution,
                        input_onesided=True)
    return y


def whiten(x, segment_len=256, fft_resolution=1024):
    """
    Whitening flattens the magnitude spectrum of a signal.
    The processing is applied for overlapping segment.

    :param x: input signal
    :param segment_len: analysis and synthesis window lengths
    :param fft_resolution: STFT frequency resolution
    :return: whitened signal
    """
    _, _, X = signal.stft(x, nperseg=segment_len,
                          nfft=fft_resolution,
                          return_onesided=True)

    # an eps term is added to the denominator
    # to avoid divisions by zero
    eps = 1e-6
    Y = X / abs(X + eps)

    _, y = signal.istft(Y, nperseg=segment_len,
                        nfft=fft_resolution,
                        input_onesided=True)
    y = normalize_std(y, x)
    return y


def normalize(x):
    return x / np.max(x)


def normalize_std(x, ref=None):
    """
    Normalizes a signal based on the std of a reference signal.
    The target std is considered 1 if a refence signal is not provided.

    :param x: input signal
    :param ref: reference signal
    :return: normalized signal
    """
    std_ref = np.std(ref) if ref is not None else 1
    std_x = np.std(x)
    assert std_x != 0, "Normalization of zero variance signal is not allowed"
    return x / std_x * std_ref


def adaptive_filtering(x, block_size=64):
    X = view_as_windows(x, block_size, 1)
    h = np.random.random(block_size)


# TODO: not working properly
def median_separation(audio, fs, segment_len=1024, fft_resolution=1024,
                      fp=17, fh=21):
    _, _, X = signal.stft(audio, fs,
                          nperseg=segment_len,
                          nfft=fft_resolution,
                          return_onesided=True)
    Y = np.abs(X) ** 2

    # P: vertical filtering - percussive matrix
    # H: horizontal filtering - harmonic matrix
    P = ndimage.median_filter(Y, (fp, 1))
    H = ndimage.median_filter(Y, (1, fh))

    # binary masks
    Mp = np.int8(P >= H)
    Mh = np.int8(P < H)

    # Wiener filter
    eps = 1e-5
    Hp = (P + eps / 2) / (P + H + eps)
    Hh = (H + eps / 2) / (P + H + eps)
    Xp = X ** Mp
    Xh = X * Mh

    _, percussive = signal.istft(P, fs,
                                 nperseg=segment_len,
                                 nfft=fft_resolution,
                                 input_onesided=True)

    _, harmonic = signal.istft(Xh, fs,
                               nperseg=segment_len,
                               nfft=fft_resolution,
                               input_onesided=True)
    percussive = percussive[0:len(audio)]
    harmonic = harmonic[0:len(audio)]

    return percussive, harmonic
