"""The one feature definition used by the notebooks, training, evaluation and (later) the firmware.

Audio is band-limited to 4 kHz by resampling to 8 kHz (the anti-alias filter of the resampler does the low-pass), cut to 1 s windows, and turned into a
log-mel patch of 40 bands over 50 to 4000 Hz: 256-point FFT (32 ms Hann window), hop 80 samples (10 ms), 101 frames. Each patch is referenced to its own maximum
(top 80 dB kept), so the loudness of the recorder hardly matters, as on the old patrol model. Output shape: (40, 101), float32, values in [-80, 0] dB.
"""
import numpy as np
from scipy.signal import resample_poly, stft

SR = 8000
WIN_S = 1.0
N_FFT, HOP, N_MELS, FMIN, FMAX = 256, 80, 40, 50.0, 4000.0
PRE_S = 0.25            # for impulsive sounds the peak sits 250 ms into the window, like the station's microphone trigger


def to_8k(x, sr):
    """Mono float32 audio at any rate -> 8 kHz (low-passed at 4 kHz by the resampler)."""
    if sr == SR: return x.astype("float32")
    g = int(np.gcd(sr, SR))
    return resample_poly(x, SR // g, sr // g).astype("float32")


def window(x, rule="any", rng=None, jitter=0, sr=SR):
    """One 1 s window from an 8 kHz clip. 'peak': place the loudest point 250 ms in. 'any': a random position (or the start when rng is None). Short clips are zero-padded."""
    n = int(WIN_S * sr)
    if len(x) <= n:
        return np.pad(x, (0, n - len(x)))
    if rule == "peak":
        env = np.convolve(np.abs(x), np.ones(sr // 100) / (sr // 100), mode="same")        # 10 ms smoothed envelope
        c = int(np.argmax(env)); lo = max(0, c - sr // 100)
        c = lo + int(np.argmax(np.abs(x[lo:c + sr // 100])))                      # the sample peak inside the loudest 10 ms
        start = c - int(PRE_S * sr)
    else:
        start = int(rng.integers(0, len(x) - n + 1)) if rng is not None else 0
    if jitter and rng is not None:
        start += int(rng.integers(-jitter, jitter + 1))       # the trigger fires a little early or late
    start = min(max(start, 0), len(x) - n)
    return x[start:start + n]


_MEL = None
def mel_filters():
    global _MEL
    if _MEL is None:
        import librosa
        _MEL = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX).astype("float32")   # (40, 129)
    return _MEL


def logmel(w):
    """1 s window at 8 kHz -> (40, 101) log-mel in dB, referenced to the patch maximum, floor -80 dB."""
    _, _, Z = stft(w, fs=SR, window="hann", nperseg=N_FFT, noverlap=N_FFT - HOP, boundary="zeros", padded=True, return_onesided=True)
    P = np.abs(Z) ** 2                                   # (129, frames)
    M = mel_filters() @ P
    db = 10 * np.log10(np.maximum(M, 1e-10))
    db = np.maximum(db - db.max(), -80.0)
    return db[:, :101].astype("float32")
