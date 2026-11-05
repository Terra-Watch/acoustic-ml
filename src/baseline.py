"""Floor baseline: logistic regression on the time-mean and time-std of the log-mel patch, trained on the same augmented sampler. Needs no torch.
Run: python src/baseline.py --out runs/baseline_lr"""
import argparse, json, sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).parent))
import data as D, evalkit as E

stats = lambda m: np.concatenate([m.mean(-1), m.std(-1)], -1)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=30000); ap.add_argument("--slice", type=int, default=0)
    ap.add_argument("--cache", default="data/cache8k"); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
    cache = D.Cache(a.cache, per_class=a.slice, seed=a.seed); rng = np.random.default_rng(a.seed); s = D.TrainSampler(cache)
    X, y = zip(*[s.sample(rng) for _ in range(a.n)]); X = stats(np.stack(X)); y = np.array(y)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    clf = LogisticRegression(max_iter=500, class_weight="balanced").fit((X - mu) / sd, y)
    win, feats = E.load_eval(cache, ("val", "test_in", "test_ext"), None if a.slice else Path(a.cache) / "eval_all.pkl")
    p = np.zeros((len(win), len(D.CLASSES))); p[:, clf.classes_] = clf.predict_proba((stats(feats.astype("float32")) - mu) / sd)
    res = E.report(win, p); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); json.dump(res, open(out / "eval.json", "w"), indent=1)
    print("val macro-F1 %.3f" % res["splits"]["val"]["macro_f1"], "test_ext macro-F1 %.3f" % res["splits"].get("test_ext", {}).get("macro_f1", float("nan")), "FA/h", res.get("false_alarms_per_hour", {}).get("value"))
