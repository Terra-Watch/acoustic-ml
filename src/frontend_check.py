"""Run where torch and torchaudio are installed (the VM). Checks frontend_ref.log_mel against the real EfficientAT AugmentMelSTFT path on real validation windows, and writes
runs/frontend/vectors.npz (windows as int16, expected patches, resampler taps, mel matrix) for the firmware's test vectors.   python src/frontend_check.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch, torchaudio

sys.path.insert(0, str(Path(__file__).parent))
import frontend_ref as R
from finetune import WIDTH, build
from upper_bound import windows

dev = "cuda" if torch.cuda.is_available() else "cpu"
_, mel = build("mn10_as", WIDTH["mn10_as"], dev, crop=R.N_BANDS, hop_ms=10); mel.eval()
imp = np.zeros(400, np.float32); imp[200] = 1.0
y = torchaudio.functional.resample(torch.from_numpy(imp)[None], 16000, 32000)[0].numpy().astype(np.float64)       # the resampler's impulse response
K = 15; m0 = 200; taps = np.array([[y[2 * (m0 + K // 2 - k) + p] for k in range(K)] for p in range(2)])      # taps[p, k] = response at output 2(m0 + K/2 - k) + p
idx = pd.read_csv("data/cache8k/index.csv", keep_default_na=False); sub = idx[idx.split == "val"].sample(40, random_state=1)
win, wav = windows(sub, Path("data/corpus_v1"), 0); take = np.arange(0, len(win), max(1, len(win) // 40))[:40]; wav = wav[take]
with torch.no_grad(): ref = mel(torchaudio.functional.resample(torch.from_numpy(wav).to(dev), 16000, 32000))[:, :R.N_BANDS].cpu().numpy()
mine = np.stack([R.log_mel(w, taps) for w in wav]); d = np.abs(ref - mine)
tm, _ = torchaudio.compliance.kaldi.get_mel_banks(128, 1024, 32000, 0.0, 15000.0, vtln_low=100.0, vtln_high=-500.0, vtln_warp_factor=1.0)
print("mel matrix max abs difference vs torchaudio:", float(np.abs(tm.numpy() - R.kaldi_mel_banks()[:, :512]).max()))
print("patches", mine.shape, "max abs difference vs the real front end:", float(d.max()), "mean", float(d.mean()), "(values span about", float(ref.max() - ref.min()), ")")
# stage by stage, to see where a difference starts (same torch calls as AugmentMelSTFT.forward)
x32 = torchaudio.functional.resample(torch.from_numpy(wav).to(dev), 16000, 32000)
mine32 = np.stack([R.upsample2(w.astype(np.float64), taps) for w in wav]); e32 = np.abs(x32.cpu().numpy() - mine32)
print("stage resample: max abs", float(e32.max()), "mean", float(e32.mean()), "| signal max", float(np.abs(x32.cpu().numpy()).max()), "| max abs ignoring the first and last 20 samples", float(e32[:, 20:-20].max()))
pre = torch.nn.functional.conv1d(x32.unsqueeze(1), torch.as_tensor([[[-.97, 1]]], dtype=x32.dtype, device=dev)).squeeze(1)
sp = torch.stft(pre, 1024, hop_length=320, win_length=800, center=True, normalized=False, window=torch.hann_window(800, periodic=False, device=dev), return_complex=False)
pw = (sp ** 2).sum(-1).cpu().numpy()
y64 = mine32[:, 1:] - 0.97 * mine32[:, :-1]; print("stage pre-emphasis: max abs", float(np.abs(pre.cpu().numpy() - y64).max()))
mp = []
for yy in y64:
    z = np.pad(yy, (512, 512), mode="reflect"); win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(800) / 799); fr = []
    for t in range(1 + (len(z) - 1024) // 320):
        seg = np.zeros(1024); seg[112:912] = z[t * 320:t * 320 + 1024][112:912] * win; sc = np.fft.rfft(seg); fr.append(sc.real ** 2 + sc.imag ** 2)
    mp.append(np.stack(fr, 1))
mp = np.stack(mp); print("stage power spectrum: max abs", float(np.abs(pw - mp).max()), "relative to the largest power", float(np.abs(pw - mp).max() / pw.max()), "| shapes", pw.shape, mp.shape)
out = Path("runs/frontend"); out.mkdir(parents=True, exist_ok=True)
np.savez(out / "vectors.npz", wav_int16=np.clip(np.round(wav * 32767), -32768, 32767).astype(np.int16), wav_float=wav, patches=ref, taps=taps, mel_basis=R.kaldi_mel_banks()[:R.N_BANDS])
print("taps", taps.shape, taps[0].round(5), taps[1].round(5))
