"""Independent check of the Belize 'background' clips the fine-tuned model alarms on: tag each clip with the PANNs AudioSet tagger (527 sound classes) and compare the clips the model alarmed on with quiet ones.
Run on a VM: python src/tag_clips.py ids.csv   (ids.csv: id, set). Writes runs/belize_audit.csv and prints a summary."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from panns_inference import AudioTagging, labels
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).parent))
from upper_bound import load16

ids = pd.read_csv(sys.argv[1]); dev = "cuda" if torch.cuda.is_available() else "cpu"; at = AudioTagging(checkpoint_path=None, device=dev)
GROUPS = {"machinery": ["Chainsaw", "Engine", "Motor vehicle (road)", "Vehicle", "Power tool", "Sawing", "Medium engine (mid frequency)", "Light engine (high frequency)", "Heavy engine (low frequency)", "Idling", "Helicopter", "Aircraft", "Motorboat, speedboat", "Lawn mower", "Mechanical fan", "Drill", "Tools", "Car", "Truck", "Motorcycle", "Boat"],
          "gunshot": ["Gunshot, gunfire", "Machine gun", "Fusillade", "Explosion", "Cap gun", "Artillery fire"],
          "insects": ["Insect", "Cricket", "Mosquito", "Fly, housefly", "Bee, wasp, etc.", "Buzz", "Cicada"], "frogs_birds": ["Frog", "Croak", "Bird", "Bird vocalization, bird call, bird song", "Chirp, tweet"],
          "rain_wind": ["Rain", "Raindrop", "Rain on surface", "Wind", "Rustling leaves", "Thunder", "Water", "Stream"]}
rows = []
for r in ids.itertuples():
    x = resample_poly(load16(Path("data/corpus_v1/test_ext") / f"{r.id}.flac"), 2, 1).astype("float32")
    clip = at.inference(x[None, :])[0][0]
    top = np.argsort(clip)[::-1][:3]
    rows.append(dict(id=r.id, set=r.set, top1=labels[top[0]], top1_p=round(float(clip[top[0]]), 2), top3="; ".join(f"{labels[i]} {clip[i]:.2f}" for i in top),
                     **{g: round(float(max(clip[labels.index(n)] for n in names if n in labels)), 3) for g, names in GROUPS.items()}))
d = pd.DataFrame(rows); Path("runs").mkdir(exist_ok=True); d.to_csv("runs/belize_audit.csv", index=False)
for g in GROUPS: print(g, "share of clips with tag probability >= 0.3:", d.groupby("set")[g].apply(lambda v: round(float((v >= 0.3).mean()), 2)).to_dict(), flush=True)
print(d.groupby("set").top1.apply(lambda v: v.value_counts().head(6).to_dict()).to_string(), flush=True)
