"""
Turns a heart sound recording into a log-mel spectrogram: a picture of
which frequencies are loud at each moment. The CNN looks at this picture.

web/js/features.js implements exactly the same steps in JavaScript, and
tests/parity check that both give the same numbers.
"""

import numpy as np
from scipy.io import wavfile

SAMPLE_RATE = 4000      # CirCor recordings are sampled at 4 kHz
N_FFT = 256             # 64 ms analysis frame
HOP = 64                # 16 ms between frames
N_MELS = 32
F_MIN = 25.0            # heart sounds and murmurs live roughly in 25-800 Hz
F_MAX = 1000.0
WINDOW_FRAMES = 247     # frames in one 4-second window
WINDOW_STEP = 125       # 2 seconds between window starts
EPS = 1e-6


def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_filterbank():
    """Triangular filters, evenly spaced on the mel scale (how humans hear pitch)."""
    n_bins = N_FFT // 2 + 1
    bin_freqs = np.arange(n_bins) * SAMPLE_RATE / N_FFT
    mel_points = np.linspace(hz_to_mel(F_MIN), hz_to_mel(F_MAX), N_MELS + 2)
    hz_points = mel_to_hz(mel_points)
    fb = np.zeros((N_MELS, n_bins))
    for m in range(N_MELS):
        left, center, right = hz_points[m], hz_points[m + 1], hz_points[m + 2]
        rising = (bin_freqs - left) / (center - left)
        falling = (right - bin_freqs) / (right - center)
        fb[m] = np.maximum(0.0, np.minimum(rising, falling))
    return fb


MEL_FB = mel_filterbank()
HANN = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)


def load_wav(path):
    rate, x = wavfile.read(path)
    if rate != SAMPLE_RATE:
        raise ValueError(f"{path}: expected {SAMPLE_RATE} Hz, got {rate}")
    if x.dtype == np.int16:
        x = x.astype(np.float64) / 32768.0
    return x.astype(np.float64)


def normalize(x):
    """Remove the DC offset and scale to unit loudness, so a quiet and a loud
    stethoscope give the same spectrogram."""
    x = x - x.mean()
    return x / (x.std() + 1e-8)


def log_mel(x):
    """Full-recording log-mel spectrogram, shape (N_MELS, n_frames)."""
    x = normalize(x)
    if len(x) < N_FFT:
        x = np.pad(x, (0, N_FFT - len(x)))
    n_frames = 1 + (len(x) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n_frames)[:, None]
    frames = x[idx] * HANN
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    return np.log10(MEL_FB @ power.T + EPS).astype(np.float32)


def window_starts(n_frames):
    """Start frame of every 4-second window, 2 seconds apart."""
    if n_frames <= WINDOW_FRAMES:
        return [0]
    return list(range(0, n_frames - WINDOW_FRAMES + 1, WINDOW_STEP))


def cut_window(spec, start):
    """One 4-second window. Short recordings are repeated to fill it."""
    if spec.shape[1] < WINDOW_FRAMES:
        reps = int(np.ceil(WINDOW_FRAMES / spec.shape[1]))
        spec = np.tile(spec, (1, reps))
    return spec[:, start:start + WINDOW_FRAMES]
