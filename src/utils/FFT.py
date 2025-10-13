import numpy as np
from scipy.fftpack import fft, fftfreq

def cal_period(seqs: np.ndarray, top_k_seasons=3):
    fft_series = fft(seqs)
    power = np.abs(fft_series)
    sample_freq = fftfreq(fft_series.size)

    pos_mask = np.where(sample_freq > 0)
    freqs = sample_freq[pos_mask]
    powers = power[pos_mask]

    # top K=3 index
    top_k_idxs = np.argpartition(powers, -top_k_seasons)[-top_k_seasons:]
    fft_periods = (1 / freqs[top_k_idxs]).astype(int)

    return fft_periods