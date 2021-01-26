import numpy as np
from numpy.lib.stride_tricks import as_strided


def strided_windowing(x, win_len, hopsize):
    # TODO: modify x with zero padding to fit the striding
    # assert (len(x) - win_len) % hopsize == 0, "size mismatch"
    padding = (len(x) - win_len) % hopsize
    y = x.copy()
    y = np.pad(y, (0, padding), 'constant')

    blocksize = y.strides[0]
    X = as_strided(y, (win_len, ((len(y) - win_len) // hopsize) + 1),
                   (blocksize, hopsize * blocksize))
    return X


def db(x):
    return 20 * np.log10(x) if x != 0 else float('-inf')
