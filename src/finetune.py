"""Fine-tune a pretrained AudioSet MobileNetV3 (EfficientAT, Schmid et al.) on our six classes, on 16 kHz audio with the same sampler and augmentations as the students.
The epoch is picked on validation macro-F1; evaluate on val, test_in and test_ext with evalkit.report (thresholds from val), so the result is directly comparable to the students.
Needs the EfficientAT repo (git clone https://github.com/fschmid56/EfficientAT ~/EfficientAT).
Run: python src/finetune.py --model mn04_as [--width 0.4] [--epochs 20] [--out runs/ft_mn04_as]"""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as Fn, torchaudio

SRC = Path(__file__).resolve().parent; sys.path.insert(0, str(SRC))
import data as D, evalkit as E, features as F
from upper_bound import SR, load16, windows

WIDTH = {"mn01_as": 0.1, "mn02_as": 0.2, "mn04_as": 0.4, "mn05_as": 0.5, "mn10_as": 1.0}


def eq16(w, rng, db=4.0):
    f = np.fft.rfftfreq(len(w), 1 / SR); pts = np.geomspace(100, 7500, 6); g = rng.normal(0, db, 6)       # random smooth response: different microphones and enclosures
    return np.fft.irfft(np.fft.rfft(w) * 10 ** (np.interp(np.log(np.maximum(f, 1)), np.log(pts), g) / 20), n=len(w)).astype(np.float32)


class Stream(torch.utils.data.IterableDataset):
    """(class uniformly, source uniformly within the class, item uniformly within the source); event mixed into a background clip with probability p_mix."""
    def __init__(self, clips, meta, seed, n, p_mix, snr, gain_db=0.0):
        self.clips, self.seed, self.n, self.p_mix, self.snr, self.epoch, self.gain_db = clips, seed, n, p_mix, snr, 0, gain_db
        self.rule = meta.rule.to_numpy(); self.by = {c: [g.index.to_numpy() for _, g in meta[meta.cls == c].groupby("source")] for c in D.CLASSES}
        self.bg = meta[meta.cls == "background"].index.to_numpy()
    def __iter__(self):
        wi = torch.utils.data.get_worker_info(); wid, nw = (wi.id, wi.num_workers) if wi else (0, 1); rng = np.random.default_rng((self.seed, self.epoch, wid))
        for _ in range(self.n // nw):
            c = D.CLASSES[int(rng.integers(len(D.CLASSES)))]; srcs = self.by[c]; items = srcs[int(rng.integers(len(srcs)))]; i = int(items[int(rng.integers(len(items)))])
            w = F.window(self.clips[i], self.rule[i], rng, jitter=int(0.08 * SR) if self.rule[i] == "peak" else 0, sr=SR)
            if c != "background" and rng.random() < self.p_mix:
                b = F.window(self.clips[int(rng.choice(self.bg))], "any", rng, sr=SR); w = D.mix_into(w, b, float(rng.uniform(*self.snr)))
            if rng.random() < 0.7: w = eq16(w, rng)
            if self.gain_db: w = w * np.float32(10 ** (rng.uniform(-self.gain_db, self.gain_db) / 20))      # a station's microphone gain and distance are unknown: the level must not decide the class
            yield torch.from_numpy(w.astype(np.float32)), D.C2I[c]


def build(name, width, dev, n_mels=128, mel_sr=32000, crop=0, hop_ms=10):
    # EfficientAT's `models` package has no __init__.py, so our src/models.py would win the import: look in its folder only.
    # It also reads metadata/ and saves weights to resources/ relative to the working directory, so run from its folder.
    eff = os.path.expanduser(os.environ.get("EFFICIENTAT", "~/EfficientAT")); saved, cwd = sys.path[:], os.getcwd()
    sys.path[:] = [eff] + [p for p in saved if Path(p or ".").resolve() != SRC]; os.chdir(eff)
    try:
        from models.mn.model import get_model
        from models.preprocess import AugmentMelSTFT
        net = get_model(num_classes=len(D.CLASSES), pretrained_name=name, width_mult=width, input_dim_f=crop or n_mels, input_dim_t=1000 // hop_ms).to(dev)
        mel = AugmentMelSTFT(n_mels=n_mels, sr=mel_sr, win_length=mel_sr // 40, hopsize=mel_sr * hop_ms // 1000, n_fft=1024 if mel_sr == 32000 else 512, freqm=max(1, n_mels // 10), timem=10).to(dev)   # 32 kHz: 800/320/1024 as pretrained; 16 kHz: 400/160/512, the front end the node computes
        mel.tw_sr = mel_sr; mel.tw_crop = crop      # crop: keep only the lowest mel bands (the 32 kHz layout's bands above 8 kHz are empty for 16 kHz audio)
    finally: sys.path[:] = saved; os.chdir(cwd)
    return net, mel


def macs_of(net, dev, shape=(1, 1, 128, 100)):   # pass the model's own (1, 1, mel bands, 100)
    """Multiply-accumulates of one forward pass (convolutions and linear layers): the board budget is counted in these."""
    tot = [0]
    def conv(m, i, o): tot[0] += o.shape[2] * o.shape[3] * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    def lin(m, i, o): tot[0] += m.in_features * m.out_features
    hs = [m.register_forward_hook(conv) for m in net.modules() if isinstance(m, nn.Conv2d)] + [m.register_forward_hook(lin) for m in net.modules() if isinstance(m, nn.Linear)]
    net.eval()
    with torch.no_grad(): net(torch.zeros(shape, device=dev))
    [h.remove() for h in hs]; return tot[0]


def logits(net, mel, x):
    if getattr(mel, "tw_sr", 32000) != SR: x = torchaudio.functional.resample(x, SR, mel.tw_sr)      # the pretrained mel front end is defined at 32 kHz; at 16 kHz the audio goes in as it is
    m = mel(x)
    if getattr(mel, "tw_crop", 0): m = m[:, :mel.tw_crop]
    return net(m.unsqueeze(1))[0]


def probs_of(net, mel, wav, dev, bs=256):
    net.eval(); mel.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(wav), bs): out.append(torch.softmax(logits(net, mel, torch.from_numpy(wav[i:i + bs]).to(dev)), 1).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="mn04_as"); ap.add_argument("--width", type=float); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--steps", type=int, default=250); ap.add_argument("--batch", type=int, default=64); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--p_mix", type=float, default=0.5); ap.add_argument("--snr_lo", type=float, default=5.0); ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--eval_only", action="store_true"); ap.add_argument("--n_mels", type=int, default=128); ap.add_argument("--crop", type=int, default=0); ap.add_argument("--hop_ms", type=int, default=10); ap.add_argument("--mel_sr", type=int, default=32000); ap.add_argument("--gain_db", type=float, default=0.0); ap.add_argument("--teacher"); ap.add_argument("--temp", type=float, default=2.0); ap.add_argument("--alpha", type=float, default=0.5)   # score the saved best.pt again and save per-window predictions
    ap.add_argument("--cache", default="data/cache8k"); ap.add_argument("--corpus", default="data/corpus_v1"); ap.add_argument("--out")
    a = ap.parse_args(); out = Path(a.out or f"runs/ft_{a.model}"); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed); dev = "cuda" if torch.cuda.is_available() else "cpu"; t0 = time.time()
    index = pd.read_csv(Path(a.cache) / "index.csv", keep_default_na=False)
    if a.limit: index = index.groupby(["split", "cls"], group_keys=False).head(a.limit)
    tr = index[index.split == "train"].reset_index(drop=True); clips = [load16(Path(a.corpus) / p) for p in tr.path]; print(len(clips), "training clips", f"({time.time() - t0:.0f} s)", flush=True)
    win, wav = windows(index[index.split != "train"], Path(a.corpus), 0); print(len(win), "evaluation windows", f"({time.time() - t0:.0f} s)", flush=True)
    net, mel = build(a.model, a.width or WIDTH[a.model], dev, a.n_mels, a.mel_sr, a.crop, a.hop_ms); params = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-2); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * a.steps, pct_start=0.15)
    nw = max(a.workers, 1); ds = Stream(clips, tr, a.seed, -(-a.steps * a.batch // nw) * nw, a.p_mix, (a.snr_lo, 20.0), a.gain_db)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, num_workers=a.workers, drop_last=True); lossf = nn.CrossEntropyLoss(label_smoothing=0.05)
    mv = (win.split == "val").to_numpy(); yv = win[mv].cls.map(D.C2I).to_numpy(); best, log = -1, []; print(f"{a.model}: {params} parameters, {macs_of(net, dev, (1, 1, a.crop or a.n_mels, 1000 // a.hop_ms)) / 1e6:.1f} M multiply-adds per 1 s window, device {dev}", flush=True)
    teacher = None
    if a.teacher:           # knowledge distillation: the fine-tuned teacher (clean input) gives soft targets
        tnet, tmel = build(a.teacher, WIDTH[a.teacher], dev); tnet.load_state_dict(torch.load(f"runs/ft_{a.teacher}/best.pt", map_location=dev)); tnet.eval(); tmel.eval(); teacher = (tnet, tmel)
    for ep in range(0 if a.eval_only else a.epochs):
        ds.epoch = ep; net.train(); mel.train(); t1 = time.time(); tot = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev); opt.zero_grad(); lg = logits(net, mel, xb); loss = lossf(lg, yb)
            if teacher is not None:
                with torch.no_grad(): tl = logits(*teacher, xb)
                loss = a.alpha * loss + (1 - a.alpha) * Fn.kl_div(Fn.log_softmax(lg / a.temp, 1), Fn.softmax(tl / a.temp, 1), reduction="batchmean") * a.temp ** 2
            loss.backward(); opt.step(); sched.step(); tot += loss.item()
        f1 = E.macro_f1(yv, probs_of(net, mel, wav[mv], dev).argmax(1)); log.append(dict(epoch=ep, loss=tot / a.steps, val_macro_f1=f1, seconds=round(time.time() - t1, 1))); print(log[-1], flush=True)
        if f1 > best: best = f1; torch.save(net.state_dict(), out / "best.pt")
    net.load_state_dict(torch.load(out / "best.pt", map_location=dev)); pr = probs_of(net, mel, wav, dev); res = E.report(win, pr); np.save(out / "probs.npy", pr.astype("float16")); win.to_csv(out / "windows.csv", index=False)   # per-window predictions, to see what the errors are
    res["run"] = str(out); res["params"] = params
    json.dump(res, open(out / "eval.json", "w"), indent=1)
    if not a.eval_only: json.dump(dict(args=vars(a), params=params, log=log), open(out / "train.json", "w"), indent=1)
    print(a.model, "val macro-F1 %.3f" % res["splits"]["val"]["macro_f1"], "FA/h", res.get("false_alarms_per_hour", {}).get("value"), flush=True)


if __name__ == "__main__": main()
