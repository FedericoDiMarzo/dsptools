from pathlib import Path
import numpy as np
import unittest
from dsp import util
from dsp.processing import denoise
from scipy.io import wavfile


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
        self.fs, self.audio = wavfile.read(Path().joinpath('tests', 'mocks', 'disco0.wav'))
        self.audio = np.sum(self.audio, axis=1)
        self.audio = self.audio[0:self.length]
        self.eps = 1e-6
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

if __name__ == '__main__':
    unittest.main()
