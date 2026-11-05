"""Train one student on the train split, pick the epoch by validation macro-F1. Never touches test_in or test_ext (evaluate.py scores frozen checkpoints).
Run: python src/train.py --model student_ir --seed 0 --out runs/student_ir_s0"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
import data as D, evalkit as E, models as M

norm = lambda m: (m + 40.0) / 20.0                              # dB patch in [-80, 0] -> about [-2, 2]; the firmware must do the same


class Stream(torch.utils.data.IterableDataset):
    def __init__(self, cache, seed, n, **kw): self.cache, self.seed, self.n, self.kw, self.epoch = cache, seed, n, kw, 0
    def __iter__(self):
        wi = torch.utils.data.get_worker_info(); wid, nw = (wi.id, wi.num_workers) if wi else (0, 1)
        rng = np.random.default_rng((self.seed, self.epoch, wid)); s = D.TrainSampler(self.cache, **self.kw)
        for _ in range(self.n // nw):
            x, y = s.sample(rng); yield torch.from_numpy(norm(x))[None], y


def probs_of(model, feats, dev, bs=1024):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(feats), bs):
            x = torch.from_numpy(norm(feats[i:i + bs].astype("float32")))[:, None].to(dev); out.append(torch.softmax(model(x), 1).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="student_ir"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--steps", type=int, default=300); ap.add_argument("--batch", type=int, default=128); ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--cache", default="data/cache8k"); ap.add_argument("--out", required=True); ap.add_argument("--slice", type=int, default=0)
    ap.add_argument("--workers", type=int, default=7); ap.add_argument("--p_mix", type=float, default=0.8); ap.add_argument("--p_eq", type=float, default=0.7); ap.add_argument("--snr_lo", type=float, default=-5.0)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed); dev = "cuda" if torch.cuda.is_available() else "cpu"
    cache = D.Cache(a.cache, per_class=a.slice, seed=a.seed)
    win, feats = E.load_eval(cache, ("val", "test_in", "test_ext"), None if a.slice else Path(a.cache) / "eval_all.pkl")
    mv = (win.split == "val").to_numpy(); win, feats = win[mv].reset_index(drop=True), feats[mv]        # training only ever looks at the validation windows
    model = M.MODELS[a.model]().to(dev); params, macs = M.count(M.MODELS[a.model]())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * (a.steps), pct_start=0.15)
    nw = max(a.workers, 1); ds = Stream(cache, a.seed, -(-a.steps * a.batch // nw) * nw, p_mix=a.p_mix, p_eq=a.p_eq, snr=(a.snr_lo, 20.0))
    dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, num_workers=a.workers, persistent_workers=False, drop_last=True)   # exactly a.steps batches per epoch: the one-cycle schedule is sized for it
    lossf = nn.CrossEntropyLoss(label_smoothing=0.05); y = win.cls.map(D.C2I).to_numpy(); best, log = -1, []
    print(f"{a.model}: {params} parameters, {macs / 1e6:.1f} M MACs, device {dev}, {len(win)} validation windows", flush=True)
    for ep in range(a.epochs):
        ds.epoch = ep; model.train(); t0 = time.time(); tot = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev); opt.zero_grad(); loss = lossf(model(xb), yb); loss.backward(); opt.step(); sched.step(); tot += loss.item()
        p = probs_of(model, feats, dev); f1 = E.macro_f1(y, p.argmax(1))
        log.append(dict(epoch=ep, loss=tot / a.steps, val_macro_f1=f1, seconds=round(time.time() - t0, 1)))
        print(log[-1], flush=True)
        if f1 > best: best = f1; torch.save(model.state_dict(), out / "best.pt")
    torch.save(model.state_dict(), out / "last.pt")
    json.dump(dict(args=vars(a), params=params, macs=macs, best_val_macro_f1=best, log=log), open(out / "train.json", "w"), indent=1)


if __name__ == "__main__": main()
