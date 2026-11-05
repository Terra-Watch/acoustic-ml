"""Decode every packed clip once, band-limit to 4 kHz (resample to 8 kHz) and store them as one int16 array: data/cache8k/audio.npy + index.csv (corpus.csv plus offset and length).
Training then memory-maps one file instead of decoding 21k FLACs per epoch. Run: .../ml/venv/bin/python -I scripts/21_cache_8k.py"""
import csv, sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "src"))
import features as F

CORPUS = HERE / "data" / "corpus_v1"
OUT = HERE / "data" / "cache8k"


def load(rel):
    x, sr = sf.read(str(CORPUS / rel), dtype="float32", always_2d=True)
    y = F.to_8k(x.mean(axis=1), sr)
    return np.clip(np.round(y * 32767), -32768, 32767).astype("int16")


if __name__ == "__main__":
    rows = list(csv.DictReader(open(CORPUS / "corpus.csv")))
    with Pool(8) as p: clips = p.map(load, [r["path"] for r in rows], chunksize=64)
    lens = np.array([len(c) for c in clips]); offs = np.concatenate([[0], np.cumsum(lens)[:-1]])
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "audio.npy", np.concatenate(clips))
    with open(OUT / "index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) + ["offset", "length"]); w.writeheader()
        for r, o, n in zip(rows, offs, lens): w.writerow({**r, "offset": int(o), "length": int(n)})
    print(len(rows), "clips,", round(lens.sum() / 8000 / 3600, 1), "h at 8 kHz,", round(lens.sum() * 2 / 1e9, 2), "GB")
