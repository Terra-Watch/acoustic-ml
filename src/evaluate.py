"""Score frozen checkpoints on val, test_in and test_ext (thresholds from val only) and write one JSON per run.
Run: python src/evaluate.py runs/student_ir_s0 [runs/...]"""
import json, sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import data as D, evalkit as E, models as M
from train import probs_of

if __name__ == "__main__":
    cache = D.Cache("data/cache8k"); win, feats = E.load_eval(cache, ("val", "test_in", "test_ext"), Path("data/cache8k") / "eval_all.pkl")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for run in sys.argv[1:]:
        run = Path(run); cfg = json.load(open(run / "train.json"))["args"]; model = M.MODELS[cfg["model"]]().to(dev)
        model.load_state_dict(torch.load(run / "best.pt", map_location=dev)); p = probs_of(model, feats, dev)
        res = E.report(win, p); res["run"] = str(run); import numpy as np; np.save(run / "probs.npy", p.astype("float16")); win.to_csv(run / "windows.csv", index=False); json.dump(res, open(run / "eval.json", "w"), indent=1)
        print(run, "val macro-F1 %.3f" % res["splits"]["val"]["macro_f1"], "FA/h", res.get("false_alarms_per_hour", {}).get("value"))
