"""Model-agnostic evaluation from window-level class probabilities: per-class precision / recall / F1, per-source recall, alert thresholds fixed on validation, false alarms per hour on held-out
background, and group-bootstrap intervals."""
import numpy as np
import pandas as pd

from data import CLASSES, C2I

ALERT = ["chainsaw", "blast"]


def prf(y, p, k=len(CLASSES)):
    out = {}
    for c in range(k):
        tp = int(((y == c) & (p == c)).sum()); fp = int(((y != c) & (p == c)).sum()); fn = int(((y == c) & (p != c)).sum())
        pr = tp / (tp + fp) if tp + fp else float("nan"); re = tp / (tp + fn) if tp + fn else float("nan")
        out[CLASSES[c]] = dict(precision=pr, recall=re, f1=(2 * pr * re / (pr + re)) if pr == pr and re == re and pr + re else float("nan"), n=int((y == c).sum()))
    return out


def macro_f1(y, p):
    f = [v["f1"] for v in prf(y, p).values() if v["n"] > 0 and v["f1"] == v["f1"]]
    return float(np.mean(f)) if f else float("nan")


def thresholds_from_val(win, probs, recall=0.8):
    """Per alert class: the probability above which `recall` of the validation windows of that class are caught."""
    th = {}
    for c in ALERT:
        m = (win.split == "val") & (win.cls == c)
        th[c] = float(np.quantile(probs[m.to_numpy(), C2I[c]], 1 - recall)) if m.any() else 1.0
    return th


def false_alarms_per_hour(win, probs, th, split="test_ext"):
    """Background windows of the held-out sources that would raise an alert (chainsaw or blast above its threshold); windows are non-overlapping 1 s."""
    m = ((win.split == split) & (win.cls == "background")).to_numpy()
    if not m.any(): return float("nan"), 0.0
    alarm = np.zeros(m.sum(), bool)
    for c in ALERT: alarm |= probs[m, C2I[c]] >= th[c]
    return float(alarm.sum() / (m.sum() / 3600)), m.sum() / 3600


def bootstrap(win, stat, n=300, seed=0):
    """95% interval of stat(sub_frame_indices) with groups resampled."""
    rng = np.random.default_rng(seed); groups = win.group.unique(); by = {g: np.flatnonzero((win.group == g).to_numpy()) for g in groups}
    vals = []
    for _ in range(n):
        ids = np.concatenate([by[g] for g in rng.choice(groups, len(groups))])
        v = stat(ids)
        if v == v: vals.append(v)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (float("nan"), float("nan"))


def report(win, probs, recall=0.8):
    """win: frame from data.eval_windows (all of val, test_in, test_ext); probs: (n, 6)."""
    y = win.cls.map(C2I).to_numpy(); pred = probs.argmax(1); res = {"splits": {}}
    th = thresholds_from_val(win, probs, recall); res["thresholds"] = th
    for s in ("val", "test_in", "test_ext"):
        m = (win.split == s).to_numpy()
        if not m.any(): continue
        sub = win[m].reset_index(drop=True); ys, ps = y[m], pred[m]
        d = {"windows": int(m.sum()), "accuracy": float((ys == ps).mean()), "macro_f1": macro_f1(ys, ps), "per_class": prf(ys, ps)}
        d["macro_f1_ci"] = bootstrap(sub, lambda ids: macro_f1(ys[ids], ps[ids]))
        d["recall_by_source"] = {f"{c}/{src}": float((ps[(sub.cls == c).to_numpy() & (sub.source == src).to_numpy()] == C2I[c]).mean())
                                 for c in CLASSES for src in sorted(sub[sub.cls == c].source.unique())}
        res["splits"][s] = d
    fa, hours = false_alarms_per_hour(win, probs, th)
    m = ((win.split == "test_ext") & (win.cls == "background")).to_numpy()
    if m.any():
        sub = win[m].reset_index(drop=True); pm = probs[m]
        def stat(ids):
            a = np.zeros(len(ids), bool)
            for c in ALERT: a |= pm[ids, C2I[c]] >= th[c]
            return a.sum() / (len(ids) / 3600)
        res["false_alarms_per_hour"] = {"value": fa, "hours": hours, "ci95": bootstrap(sub, stat)}
    res["alert_recall_at_threshold"] = {}
    for s in ("test_in", "test_ext"):
        for c in ALERT:
            m = ((win.split == s) & (win.cls == c)).to_numpy()
            if m.any(): res["alert_recall_at_threshold"][f"{s}/{c}"] = float((probs[m, C2I[c]] >= th[c]).mean())
    return res


def load_eval(cache, splits=("val", "test_in", "test_ext"), path=None):
    """Windows and log-mel features for the evaluation splits; stored once next to the cache because computing them takes minutes."""
    import pickle
    from pathlib import Path
    import data as D
    if path and Path(path).exists():
        with open(path, "rb") as f: return pickle.load(f)
    out = D.eval_windows(cache, list(splits))
    if path:
        with open(path, "wb") as f: pickle.dump(out, f)
    return out
