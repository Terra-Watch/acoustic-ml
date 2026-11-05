"""Continuous test: slide a 1 s window (hop 0.5 s) through every held-out clip of 5 s or more that is chainsaw or background, and save the class probabilities of each window.
The station runs the classifier on a stream, so alert logic that needs several windows in a row (chainsaw) can only be judged on continuous audio.
Run on the VM that holds the fine-tuned model: python src/sliding.py --model mn10_as   (reads runs/ft_<model>/best.pt, writes runs/ft_<model>/sliding.npz)"""
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
import data as D
from finetune import WIDTH, build, probs_of
from upper_bound import SR, load16

ap = argparse.ArgumentParser(); ap.add_argument("--model", default="mn10_as"); ap.add_argument("--hop", type=float, default=0.5)
ap.add_argument("--cache", default="data/cache8k"); ap.add_argument("--corpus", default="data/corpus_v1"); a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"; run = Path("runs") / f"ft_{a.model}"
idx = pd.read_csv(Path(a.cache) / "index.csv", keep_default_na=False)
idx = idx[(idx.split != "train") & (idx.dur.astype(float) >= 5) & idx.cls.isin(["chainsaw", "background"])].reset_index(drop=True)
net, mel = build(a.model, WIDTH[a.model], dev); net.load_state_dict(torch.load(run / "best.pt", map_location=dev)); print(len(idx), "clips", flush=True)
probs, meta, hop = [], [], int(a.hop * SR)
for i, r in idx.iterrows():
    x = load16(Path(a.corpus) / r.path)
    if len(x) < SR: continue
    starts = np.arange(0, len(x) - SR + 1, hop); w = np.stack([x[s:s + SR] for s in starts]).astype("float32")
    p = probs_of(net, mel, w, dev); probs.append(p.astype("float16")); meta += [(r.id, r.cls, r.source, r.split, s / SR) for s in starts]
    if i % 100 == 0: print(i, len(meta), flush=True)
pd.DataFrame(meta, columns=["clip", "cls", "source", "split", "t"]).to_csv(run / "sliding.csv", index=False); np.save(run / "sliding.npy", np.concatenate(probs)); print("windows", len(meta), flush=True)
