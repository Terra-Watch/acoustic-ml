"""Onset-triggered test for the single-event class (blast): the station only classifies a window that follows a sharp onset, placed like the gunshot windows the model was trained on (loudest point 250 ms in).
Here every held-out blast or background clip is treated as one trigger, at its loudest point, which is the worst case for background (every clip would trigger). Saves the probabilities.
Run on the VM that holds the fine-tuned model: python src/triggered.py --model mn10_as   (writes runs/ft_<model>/triggered.csv and .npy)"""
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
import features as F
from finetune import WIDTH, build, probs_of
from upper_bound import SR, load16

ap = argparse.ArgumentParser(); ap.add_argument("--model", default="mn10_as"); ap.add_argument("--cache", default="data/cache8k"); ap.add_argument("--corpus", default="data/corpus_v1"); a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"; run = Path("runs") / f"ft_{a.model}"
idx = pd.read_csv(Path(a.cache) / "index.csv", keep_default_na=False)
idx = idx[(idx.split != "train") & idx.cls.isin(["blast", "background"])].reset_index(drop=True)
net, mel = build(a.model, WIDTH[a.model], dev); net.load_state_dict(torch.load(run / "best.pt", map_location=dev)); print(len(idx), "clips", flush=True)
wav = np.stack([F.window(load16(Path(a.corpus) / r.path), "peak", sr=SR) for r in idx.itertuples()]).astype("float32")
np.save(run / "triggered.npy", probs_of(net, mel, wav, dev).astype("float16")); idx[["id", "cls", "source", "split", "group"]].to_csv(run / "triggered.csv", index=False); print("done", flush=True)
