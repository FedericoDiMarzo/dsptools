import numpy as np
from scipy import fft, signal, ndimage


def normalize(x):
    """
    Normalizes a signal in the range -1 1.

    :param x: input signal
    :return: normalized signal
    """
    normalization_factor = np.max(x) - np.min(x)
    assert normalization_factor != 0, "Normalization of all zero valued signal is not allowed"
    # newvalue= (max'-min')/(max-min)*(oldvalue-min)+min'
    return 2 / normalization_factor * (x - np.max(x)) + 1


def normalize_std(x, ref):
    """
    Normalizes a signal based on the std of a reference signal.

    :param x: input signal
    :param ref: reference signal
    :return: normalized signal
    """
    std_ref = np.std(ref)
    std_x = np.std(x)
    assert std_x != 0, "Normalization of zero variance signal is not allowed"
    return x / std_x * std_ref


def denoise(x, noise, segment_len=256, fft_resolution=1024):
    """
    Applies a Weiner filter to the input to reduce the noise.

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
    The processing is applied for all the overlapping segments.

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


def hpss(audio, fs, temporal_kernel=0.25, frequency_kernel=1500):
    """
    Harmonic Percussive Sound Separation using a vertical and
    horizontal median filtering in the STFT

    :param audio: input signal
    :param fs: sampling frequency
    :param temporal_kernel: horizontal median filter order
    :param frequency_kernel: vertical median filter order
    :return: harmonic, percussive
    """
    segment_len = 512
    fft_resolution = 1024
    kernel_size = (
        int(np.ceil(temporal_kernel * fs / (segment_len / 2))),  # horizontal filter order
        int(np.ceil(frequency_kernel * fft_resolution / fs)),  # vertical filter order
    )
    _, _, X = signal.stft(audio, fs,
                          nperseg=segment_len,
                          nfft=fft_resolution,
                          return_onesided=True)
    # phase = np.exp(1j*np.angle(X))
    X_mag = np.abs(X)

    # H: horizontal filtering - harmonic matrix
    # P: vertical filtering - percussive matrix
    H = ndimage.median_filter(X_mag, (1, kernel_size[0])) ** 2
    P = ndimage.median_filter(X_mag, (kernel_size[1], 1)) ** 2

    # Wiener filter
    eps = 1e-10
    mask_h = (H + eps / 2) / (P + H + eps)
    mask_p = (P + eps / 2) / (P + H + eps)
    Xh = X * mask_h
    Xp = X * mask_p

    _, harmonic = signal.istft(Xh, fs,
                               nperseg=segment_len,
                               nfft=fft_resolution,
                               input_onesided=True)

    _, percussive = signal.istft(Xp, fs,
                                 nperseg=segment_len,
                                 nfft=fft_resolution,
                                 input_onesided=True)

    percussive = percussive[0:len(audio)]
    harmonic = harmonic[0:len(audio)]

    return harmonic, percussive
