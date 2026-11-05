"""Unit tests for the feature definition and the evaluation kit (numpy only). Run: .../ml/venv/bin/python -I -m pytest tests -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import data as D, evalkit as E, features as F


def test_logmel_shape_range_and_gain_invariance():
    x = np.random.default_rng(0).standard_normal(8000).astype("float32") * 0.1
    a, b = F.logmel(x), F.logmel(x * 7.0)
    assert a.shape == (40, 101) and a.max() == 0 and a.min() >= -80
    assert np.allclose(a, b, atol=1e-3)                          # referenced to its own maximum: loudness drops out


def test_peak_window_puts_the_loudest_point_250ms_in():
    x = np.zeros(8 * 8000, "float32"); x[5 * 8000 + 100] = 1.0
    w = F.window(x, "peak"); assert abs(int(np.argmax(np.abs(w))) - int(0.25 * 8000)) < 20
    short = F.window(np.ones(3000, "float32"), "peak"); assert len(short) == 8000


def test_resampling_removes_everything_above_4khz():
    sr = 16000; t = np.arange(sr) / sr
    hi = np.sin(2 * np.pi * 6000 * t).astype("float32"); lo = np.sin(2 * np.pi * 1000 * t).astype("float32")
    assert np.abs(F.to_8k(hi, sr)[200:-200]).max() < 0.05 and np.abs(F.to_8k(lo, sr)[200:-200]).max() > 0.9


def test_mix_hits_the_requested_snr():
    rng = np.random.default_rng(1); ev = rng.standard_normal(8000).astype("float32"); bg = rng.standard_normal(8000).astype("float32") * 0.3
    m = D.mix_into(ev, bg, 10.0); resid = m - bg
    assert abs(20 * np.log10(D.rms(resid) / D.rms(bg)) - 10.0) < 0.1


def _win(n_bg=3600, n_ev=100):
    rows = [("val", "blast", "a", f"g{i}") for i in range(n_ev)] + [("test_ext", "background", "b", f"h{i}") for i in range(n_bg)]
    return pd.DataFrame(rows, columns=["split", "cls", "source", "group"])


def test_threshold_keeps_the_requested_recall_and_counts_false_alarms():
    win = _win(); p = np.zeros((len(win), 6)); p[:, D.C2I["background"]] = 1.0
    p[:100, D.C2I["blast"]] = np.linspace(0.5, 1.0, 100)               # validation blasts
    p[100:110, D.C2I["blast"]] = 0.99                                  # 10 of 3600 background seconds look like a blast
    th = E.thresholds_from_val(win, p, 0.8)
    assert (p[:100, D.C2I["blast"]] >= th["blast"]).mean() >= 0.8
    fa, hours = E.false_alarms_per_hour(win, p, th); assert abs(hours - 1.0) < 1e-9 and fa == 10.0


def test_report_runs_and_macro_f1_is_perfect_for_perfect_predictions():
    win = pd.DataFrame({"split": ["val"] * 6, "cls": D.CLASSES, "source": ["s"] * 6, "group": list("abcdef")}); p = np.eye(6)
    r = E.report(win, p); assert r["splits"]["val"]["macro_f1"] == 1.0
