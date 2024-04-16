import numpy as np
import pywt


def wavelet_denoising(data: np.ndarray):
    db4 = pywt.Wavelet('db4')
    coeffs = pywt.wavedec(data, db4)
    coeffs[len(coeffs)-1] *= 0
    coeffs[len(coeffs)-2] *= 0
    meta = pywt.waverec(coeffs, db4)

    if len(data) % 2 == 1:
        meta = meta[:-1]

    return meta
