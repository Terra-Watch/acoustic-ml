"""Step 1 of the int8 test: export the fine-tuned model to ONNX (one 1 s mel patch in, class scores out), and save (a) 800 real validation and test_in mel patches as the quantisation calibration set,
(b) 2,000 validation + 2,000 test_in mel patches with the float model's class scores, so the int8 model can be scored on exactly the same patches.
Run on the VM that holds the model: python src/export_onnx.py --model mn10_as --run runs/ft_mn10_as_c106h10 --crop 106 --hop_ms 10  (writes <run>/export/*)"""
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
import data as D
from finetune import WIDTH, build, logits
from upper_bound import windows

ap = argparse.ArgumentParser(); ap.add_argument("--model", default="mn10_as"); ap.add_argument("--run", required=True); ap.add_argument("--crop", type=int, default=0); ap.add_argument("--hop_ms", type=int, default=10)
ap.add_argument("--cache", default="data/cache8k"); ap.add_argument("--corpus", default="data/corpus_v1"); a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"; run = Path(a.run); out = run / "export"; out.mkdir(exist_ok=True)
net, mel = build(a.model, WIDTH[a.model], dev, crop=a.crop, hop_ms=a.hop_ms); net.load_state_dict(torch.load(run / "best.pt", map_location=dev)); net.eval(); mel.eval()

class Head(torch.nn.Module):          # the exported graph: mel patch -> class scores (the mel front end is done by the firmware)
    def __init__(self, n): super().__init__(); self.n = n
    def forward(self, x): return self.n(x)[0]

index = pd.read_csv(Path(a.cache) / "index.csv", keep_default_na=False)
rng = np.random.default_rng(0)
parts = []
for split in ("val", "test_in"):
    sub = index[index.split == split]; sub = sub.sample(min(len(sub), 1200), random_state=0)
    win, wav = windows(sub, Path(a.corpus), 0); take = rng.choice(len(win), min(2000, len(win)), replace=False); parts.append((split, win.iloc[take].reset_index(drop=True), wav[take]))
patches, labels, splits, ref = [], [], [], []
with torch.no_grad():
    for split, win, wav in parts:
        for i in range(0, len(wav), 128):
            x = torch.from_numpy(wav[i:i + 128]).to(dev)
            import torchaudio
            x = torchaudio.functional.resample(x, 16000, 32000); m = mel(x)
            if a.crop: m = m[:, :a.crop]
            patches.append(m.cpu().numpy()); ref.append(net(m.unsqueeze(1))[0].cpu().numpy())
        labels += list(win.cls.map(D.C2I)); splits += [split] * len(win)
P = np.concatenate(patches).astype("float32"); R = np.concatenate(ref).astype("float32")
np.save(out / "patches.npy", P); np.save(out / "float_logits.npy", R); pd.DataFrame({"label": labels, "split": splits}).to_csv(out / "meta.csv", index=False)
np.save(out / "calib_nhwc.npy", P[::max(1, len(P) // 800)][:800][..., None])          # NHWC for onnx2tf
torch.onnx.export(Head(net), torch.zeros(1, 1, P.shape[1], P.shape[2], device=dev), str(out / "model.onnx"), input_names=["mel"], output_names=["scores"], opset_version=13)
print("patches", P.shape, "exported", out / "model.onnx", flush=True)
