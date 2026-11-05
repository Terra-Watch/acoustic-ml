"""Upper bound: frozen embeddings of a large pretrained audio model + a linear classifier, on the same 1 s windows and splits as the students (16 kHz audio, so the 8 kHz band limit is not applied).
The result tells whether the students are limited by their size (the large model is clearly better) or by the data (it is not). Scored with evalkit.report, so it is directly comparable.
Run: python src/upper_bound.py --model panns,passt [--limit 40]   (writes runs/upper_<model>/eval.json)"""
import argparse, json, sys, time
from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).parent))
import data as D, evalkit as E, features as F

SR = 16000


def load16(path):
    x, sr = sf.read(path, dtype="float32", always_2d=True); x = x.mean(axis=1)
    if sr != SR: g = gcd(sr, SR); x = resample_poly(x, SR // g, sr // g).astype("float32")
    return x


def eval_starts(x, r):
    """The same windows as data.eval_windows (one peak-aligned window for impulsive items, else evenly spread non-overlapping 1 s windows), at 16 kHz."""
    if r.rule == "peak" or len(x) <= SR: return [F.window(x, r.rule, sr=SR)]
    k = 120 if (r.split == "test_ext" and r.cls == "background") else 16
    return [x[s:s + SR] for s in np.unique(np.linspace(0, len(x) - SR, min(k, len(x) // SR)).astype(int))]


def windows(index, root, limit):
    rng = np.random.default_rng(0); rows, wavs = [], []
    if limit: index = index.groupby(["split", "cls"], group_keys=False).head(limit)
    for _, r in index.iterrows():
        x = load16(root / r.path)
        if r.split == "train":
            ws = [F.window(x, r.rule, sr=SR)] if (r.rule == "peak" or len(x) <= SR) else [F.window(x, "any", rng, sr=SR) for _ in range(3)]   # up to 3 random windows per long training clip
        else: ws = eval_starts(x, r)
        for w in ws: rows.append((r.id, r.cls, r.source, r.group, r.split)); wavs.append(w)
    return pd.DataFrame(rows, columns=["item", "cls", "source", "group", "split"]), np.stack(wavs).astype("float32")


def embedder(name, dev):
    """Returns f(batch of 16 kHz windows, shape (n, 16000)) -> (n, d) embeddings."""
    up = lambda x: resample_poly(x, 2, 1, axis=1).astype("float32")        # both models were trained at 32 kHz
    if name == "panns":
        from panns_inference import AudioTagging
        at = AudioTagging(checkpoint_path=None, device=dev)
        return lambda x: at.inference(up(x))[1]
    if name == "passt":
        import torch
        from hear21passt.base import get_scene_embeddings, load_model
        m = load_model().to(dev).eval()
        def f(x):
            with torch.no_grad(): return get_scene_embeddings(torch.from_numpy(up(x)).to(dev), m).cpu().numpy()
        return f
    raise SystemExit(f"unknown model {name}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cache", default="data/cache8k"); ap.add_argument("--corpus", default="data/corpus_v1"); ap.add_argument("--batch", type=int, default=128)
    a = ap.parse_args()
    import torch; dev = "cuda" if torch.cuda.is_available() else "cpu"
    index = pd.read_csv(Path(a.cache) / "index.csv", keep_default_na=False); t0 = time.time()
    win, wav = windows(index, Path(a.corpus), a.limit); print(f"{len(win)} windows ({time.time() - t0:.0f} s)", flush=True)
    for name in a.model.split(","):          # the windows are built once and shared by the models
        f = embedder(name, dev); t1 = time.time()
        emb = np.concatenate([f(wav[i:i + a.batch]) for i in range(0, len(wav), a.batch)]).astype("float32"); print(name, "embeddings", emb.shape, f"({time.time() - t1:.0f} s)", flush=True)
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        tr = (win.split == "train").to_numpy(); y = win.cls.map(D.C2I).to_numpy(); sc = StandardScaler().fit(emb[tr])
        clf = LogisticRegression(max_iter=300, class_weight="balanced").fit(sc.transform(emb[tr]), y[tr])      # linear probe; the encoder stays frozen
        ev = (win.split != "train").to_numpy(); win_ev = win[ev].reset_index(drop=True)
        probs = np.zeros((ev.sum(), len(D.CLASSES))); probs[:, clf.classes_] = clf.predict_proba(sc.transform(emb[ev]))
        res = E.report(win_ev, probs); res["run"] = f"upper_{name}"; out = Path("runs") / f"upper_{name}"; out.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(out / "eval.json", "w"), indent=1)
        print(name, "val macro-F1 %.3f" % res["splits"]["val"]["macro_f1"], "FA/h", res.get("false_alarms_per_hour", {}).get("value"), flush=True)


if __name__ == "__main__": main()
