"""Reference for the node's audio front end, in plain numpy so that the firmware can be written line by line against it and checked with test vectors.
It reproduces what the trained mn10_as saw: 16 kHz audio (1 s) -> torchaudio sinc resample to 32 kHz -> pre-emphasis -> STFT (1024 point, 800-sample Hann window, hop 320, reflect-padded centre)
-> power -> the first 106 of EfficientAT's 128 Kaldi mel bands (0 to 15 kHz) -> ln(x + 1e-5) -> (x + 4.5) / 5.   Output: (106, 100) float.
frontend_check.py proves this equals the real AugmentMelSTFT path on the training machine and writes the test vectors and constants for the firmware."""
import numpy as np

SR_IN, SR = 16000, 32000
N_FFT, WIN, HOP = 1024, 800, 320
N_BANDS = 106                     # the bands above about 8 kHz are empty for 16 kHz audio
FMIN, FMAX = 0.0, 15000.0


def mel_scale(f): return 1127.0 * np.log(1.0 + f / 700.0)


def kaldi_mel_banks(num_bins=128, n_fft=N_FFT, sr=SR, low=FMIN, high=FMAX):
    """torchaudio.compliance.kaldi.get_mel_banks with no VTLN warping, padded with one zero column: shape (num_bins, n_fft / 2 + 1)."""
    num_fft_bins = n_fft // 2; width = sr / n_fft
    ml, mh = mel_scale(low), mel_scale(high); delta = (mh - ml) / (num_bins + 1)
    b = np.arange(num_bins)[:, None]
    left, center, right = ml + b * delta, ml + (b + 1.0) * delta, ml + (b + 2.0) * delta
    mel = mel_scale(width * np.arange(num_fft_bins))[None, :]
    up, down = (mel - left) / (center - left), (right - mel) / (right - center)
    return np.pad(np.maximum(0.0, np.minimum(up, down)), ((0, 0), (0, 1)))


def upsample2(x, taps):
    """x: 16 kHz samples. taps: (2, K) polyphase taps of torchaudio's 16k -> 32k sinc resampler (K odd, centred): y[2n + p] = sum_k taps[p, k] * x[n + k - K // 2]. Zero beyond the ends, as torchaudio pads."""
    K = taps.shape[1]; pad = np.pad(x, (K // 2, K // 2)); y = np.zeros(2 * len(x))
    for p in range(2): y[p::2] = np.convolve(pad, taps[p][::-1], mode="valid")[:len(x)]
    return y


def log_mel(x16, taps, mel_basis=None):
    """x16: 16000 float samples in [-1, 1]. Returns the (106, 100) patch the network takes."""
    if mel_basis is None: mel_basis = kaldi_mel_banks()[:N_BANDS]
    x = upsample2(np.asarray(x16, dtype=np.float64), taps)
    y = x[1:] - 0.97 * x[:-1]                                             # conv1d with [-0.97, 1]: y[n] = x[n + 1] - 0.97 x[n]
    y = np.pad(y, (N_FFT // 2, N_FFT // 2), mode="reflect")               # centre=True
    win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(WIN) / (WIN - 1))     # torch.hann_window(800, periodic=False)
    full = np.zeros(N_FFT); lo = (N_FFT - WIN) // 2
    n_frames = 1 + (len(y) - N_FFT) // HOP; power = np.empty((N_FFT // 2 + 1, n_frames))
    for t in range(n_frames):
        seg = np.zeros(N_FFT); seg[lo:lo + WIN] = y[t * HOP:t * HOP + N_FFT][lo:lo + WIN] * win
        spec = np.fft.rfft(seg); power[:, t] = spec.real ** 2 + spec.imag ** 2
    return (np.log(mel_basis @ power + 1e-5) + 4.5) / 5.0
