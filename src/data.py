"""Corpus access, training-sample generation with augmentation, and fixed evaluation windows. numpy only (torch wrappers live in train.py)."""
from pathlib import Path

import numpy as np
import pandas as pd

import features as F

CLASSES = ["chainsaw", "blast", "fire", "wood", "wind", "background"]
C2I = {c: i for i, c in enumerate(CLASSES)}
N = int(F.SR * F.WIN_S)


class Cache:
    """data/cache8k: one int16 array of every clip at 8 kHz (memory-mapped) plus index.csv. Row i of `index` is clip i."""
    def __init__(self, root, per_class=0, seed=0):
        root = Path(root)
        idx = pd.read_csv(root / "index.csv", keep_default_na=False)
        for c in ("offset", "length", "cv_fold"): idx[c] = idx[c].astype(int)
        if per_class:                                           # debug slice: at most per_class items per class and split
            idx = idx.sample(frac=1, random_state=seed).groupby(["cls", "split"]).head(per_class)
        self.index = idx.reset_index(drop=True)
        self._offs, self._lens = self.index.offset.to_numpy(), self.index.length.to_numpy()
        self.audio = np.load(root / "audio.npy", mmap_mode="r")

    def clip(self, i):
        o, n = self._offs[i], self._lens[i]
        return np.asarray(self.audio[o:o + n], dtype=np.float32) / 32768.0


# ---- augmentation ---------------------------------------------------------------------------------------------------------------------------------
def rms(x): return float(np.sqrt(np.mean(x ** 2)) + 1e-9)


def mix_into(event, bg, snr_db):
    """Scale the event so that event_rms / bg_rms = snr_db, add both."""
    return bg + event * (rms(bg) / rms(event) * 10 ** (snr_db / 20))


def random_eq(w, rng, strength_db=4.0):
    """A random smooth frequency response (+-strength_db at 6 control points between 100 and 3800 Hz): stands in for different microphones and enclosures."""
    f = np.fft.rfftfreq(len(w), 1 / F.SR)
    pts = np.geomspace(100, 3800, 6); g = rng.normal(0, strength_db, 6)
    gain = 10 ** (np.interp(np.log(np.maximum(f, 1)), np.log(pts), g) / 20)
    return np.fft.irfft(np.fft.rfft(w) * gain, n=len(w)).astype(np.float32)


def spec_augment(m, rng):
    m = m.copy()
    if rng.random() < 0.5: t = int(rng.integers(0, 10)); t0 = int(rng.integers(0, m.shape[1] - t)); m[:, t0:t0 + t] = -80.0
    if rng.random() < 0.5: f = int(rng.integers(0, 5)); f0 = int(rng.integers(0, m.shape[0] - f)); m[f0:f0 + f, :] = -80.0
    return m


class TrainSampler:
    """Draws (class uniformly, source uniformly within the class, item uniformly within the source), so that neither the huge background class nor the largest source dominates."""
    def __init__(self, cache, split="train", p_mix=0.8, snr=(-5.0, 20.0), p_eq=0.7, class_weights=None):
        self.c = cache; idx = cache.index; self.p_mix, self.snr, self.p_eq = p_mix, snr, p_eq
        sel = idx[idx.split == split]
        self.by_class = {c: [g.index.to_numpy() for _, g in sel[sel.cls == c].groupby("source")] for c in CLASSES}
        self.classes = [c for c in CLASSES if self.by_class[c]]
        w = np.array([(class_weights or {}).get(c, 1.0) for c in self.classes]); self.cw = w / w.sum()
        self.bg = sel[sel.cls == "background"].index.to_numpy()

    def item(self, rng):
        c = self.classes[int(rng.choice(len(self.classes), p=self.cw))]
        srcs = self.by_class[c]; items = srcs[int(rng.integers(len(srcs)))]
        return c, int(items[int(rng.integers(len(items)))])

    def sample(self, rng):
        c, i = self.item(rng); row = self.c.index.iloc[i]
        w = F.window(self.c.clip(i), row.rule, rng, jitter=int(0.08 * F.SR) if row.rule == "peak" else 0)
        if len(self.bg) and ((c != "background" and rng.random() < self.p_mix) or (c == "background" and rng.random() < 0.3)):
            j = int(self.bg[int(rng.integers(len(self.bg)))])
            b = F.window(self.c.clip(j), "any", rng)
            w = mix_into(w, b, float(rng.uniform(*self.snr)) if c != "background" else float(rng.uniform(0, 15)))
        if rng.random() < self.p_eq: w = random_eq(w, rng)
        return spec_augment(F.logmel(w), rng), C2I[c]


# ---- evaluation windows -----------------------------------------------------------------------------------------------------------------------------
def eval_windows(cache, splits, max_windows=16, max_windows_bg_ext=120):
    """Fixed, augmentation-free windows. Impulsive items: one window with the peak 250 ms in. Others: non-overlapping 1 s windows spread over the clip (up to max_windows).
    Returns (frame with item/class/source/group/split, features float16 (n, 40, 101))."""
    rows, feats = [], []
    idx = cache.index
    for i in idx.index[idx.split.isin(splits)]:
        r = idx.iloc[i]; x = cache.clip(i)
        if r.rule == "peak" or len(x) <= N: starts = [None]
        else:
            k = max_windows_bg_ext if (r.split == "test_ext" and r.cls == "background") else max_windows
            starts = np.unique(np.linspace(0, len(x) - N, min(k, len(x) // N)).astype(int))
        for s in starts:
            w = F.window(x, r.rule) if s is None else x[s:s + N]
            feats.append(F.logmel(w).astype(np.float16)); rows.append((i, r.cls, r.source, r.group, r.split))
    return pd.DataFrame(rows, columns=["item", "cls", "source", "group", "split"]), np.stack(feats)
