from pathlib import Path
import numpy as np
import unittest
from dsp import util
from dsp.processing import denoise, normalize, normalize_std, whiten, hpss
from scipy.io import wavfile
from librosa import feature


class TestDb(unittest.TestCase):
    def setUp(self):
        self.eps = 1e-10

    def test_one(self):
        self.assertLess(util.db(1), self.eps)

    def test_zero(self):
        self.assertEqual(util.db(0), float('-inf'))

    def test_manual(self):
        self.assertLess(np.abs(util.db(10) - 20), self.eps)


class TestStridedWindowing(unittest.TestCase):
    def assertMEqual(self, A, B):
        self.assertTrue((A.shape == B.shape))
        self.assertTrue((A == B).all())

    def test_int_array(self):
        data = np.arange(9)
        expected = np.array([[0, 1, 2], [2, 3, 4],
                             [4, 5, 6], [6, 7, 8]]).transpose()
        result = util.strided_windowing(data, 3, 2)
        self.assertMEqual(expected, result)

    def test_float_array(self):
        data = np.arange(4).astype(float)
        expected = np.array([[0, 1], [2, 3]]).transpose()
        result = util.strided_windowing(data, 2, 2)
        self.assertMEqual(expected, result)

    def test_disjointed_windows(self):
        data = np.arange(4)
        expected = np.array([[0, 1], [2, 3]]).transpose()
        result = util.strided_windowing(data, 2, 2)
        self.assertMEqual(expected, result)

    def test_padding(self):
        data = np.arange(3)
        expected = np.array([[0, 1], [2, 0]]).transpose()
        result = util.strided_windowing(data, 2, 2)
        self.assertMEqual(expected, result)

    def test_not_copy(self):
        data = np.arange(4)
        result = util.strided_windowing(data, 2, 2)
        result[0, 0] = 10
        self.assertNotEqual(data[0], result[0, 0])


class TestDenoise(unittest.TestCase):
    def setUp(self):
        self.length = 2 ** 18
        self.noise = np.random.normal(size=self.length)
        self.fs, self.audio = wavfile.read(Path('.').joinpath('mocks', 'disco0.wav'))
        self.audio = np.sum(self.audio, axis=1)
        self.audio = self.audio[0:self.length]
        self.eps = 1e-10
        self.var_noise = np.var(self.noise)

    def assertGaussianVar(self, intensity):
        noise = self.noise * intensity
        audio_noise = self.audio + noise
        result = denoise(audio_noise, noise)
        # the variance of noise + clean should be equal to the variance of the noisy signal
        self.assertLess(np.var(audio_noise) - (self.var_noise * intensity + np.var(result)), self.eps)

    def test_low_gaussian(self):
        self.assertGaussianVar(0.1)

    def test_mid_gaussian(self):
        self.assertGaussianVar(0.3)

    def test_high_gaussian(self):
        self.assertGaussianVar(0.8)


class TestNormalize(unittest.TestCase):
    def setUp(self):
        self.x = np.random.normal(size=100)
        self.eps = 1e-10

    def assertNormalization(self, signal_to_normalize):
        normalized = normalize(signal_to_normalize)
        self.assertLess(np.min(normalized) + 1, self.eps)
        self.assertLess(np.max(normalized) - 1, self.eps)

    def test_zero_mean(self):
        self.assertNormalization(self.x * 10)

    def test_nonzero_mean(self):
        self.assertNormalization(self.x * 5 + 3)

    def test_not_copy(self):
        y = normalize(self.x)
        y[0] = 20
        self.assertNotEqual(self.x[0], y[0])

    def test_null_signal(self):
        null_signal = np.zeros(10)
        with self.assertRaises(AssertionError):
            normalize(null_signal)


class TestNormalizeStd(unittest.TestCase):
    def setUp(self):
        self.x = np.random.normal(size=100)
        self.eps = 1e-10

    def assertNormalizationStd(self, signal_to_normalize, reference):
        normalized = normalize_std(signal_to_normalize, reference)
        self.assertLess(np.std(normalized) - np.std(reference), self.eps)

    def test_zero_mean(self):
        self.assertNormalizationStd(self.x, self.x * 100)

    def test_nonzero_mean1(self):
        self.assertNormalizationStd(self.x + 10, self.x * 100)

    def test_nonzero_mean2(self):
        self.assertNormalizationStd(self.x + 10, self.x * 100 - 3)

    def test_not_copy(self):
        y = normalize_std(self.x, 4)
        y[0] = 100
        self.assertNotEqual(y[0], self.x[0])

    def test_null_signal(self):
        null_signal = np.zeros(10)
        with self.assertRaises(AssertionError):
            normalize_std(null_signal, self.x)


class TestWhiten(unittest.TestCase):
    def setUp(self):
        self.fs, self.audio = wavfile.read(Path('.').joinpath('mocks', 'disco0.wav'))
        self.noise = np.random.normal(size=10000)
        self.noise_spectral_flatness = np.mean(feature.spectral_flatness(self.noise))
        self.noise_threshold = util.db(self.noise_spectral_flatness * 0.8)
        self.audio = np.sum(self.audio, axis=1)
        self.audio = self.audio / np.max(self.audio)

    def test_on_track(self):
        whitened = whiten(self.audio)
        spectral_flatness = np.mean(feature.spectral_flatness(whitened))
        self.assertGreater(util.db(spectral_flatness), self.noise_threshold)


class TestHPSS(unittest.TestCase):
    def setUp(self):
        self.fs, self.audio = wavfile.read(Path('.').joinpath('mocks', 'disco0.wav'))
        self.audio = np.sum(self.audio, axis=1)
        self.audio = self.audio / np.max(self.audio)
        self.eps = 1e-10

    def test_sum(self):
        h, p = hpss(self.audio, self.fs)
        self.assertLess(np.mean((self.audio - p - h) ** 2), self.eps)


if __name__ == '__main__':
    unittest.main()
