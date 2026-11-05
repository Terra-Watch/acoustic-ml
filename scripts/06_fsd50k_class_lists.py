"""Write fetch lists for more FSD50K classes (metadata already downloaded by script 01).

Each list row has the columns script 03 expects (fname, split, bucket); `bucket` holds the FSD50K label so one
fetch can take several classes: python3 -I scripts/03_fsd50k_fetch.py manifests/fsd50k_classes.csv Wood,Wind
Clips keep every licence; the licence column decides later what goes into the open-licence variant.
Run: python3 -I scripts/06_fsd50k_class_lists.py
"""
import csv, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data" / "fsd50k"
LABELS = ["Wood", "Wind", "Rain", "Sawing", "Gunshot_and_gunfire", "Fireworks"]
info = {}
for f in ("dev_clips_info_FSD50K.json", "eval_clips_info_FSD50K.json"):
    info.update(json.loads((D / f).read_text()))


def lic(u):
    if "zero" in u or "publicdomain" in u: return "CC0"
    return "BY-NC" if "by-nc" in u else "Sampling+" if "sampling" in u else "BY"


rows = []
for split in ("dev", "eval"):
    for r in csv.DictReader(open(D / f"{split}.csv")):
        for l in r["labels"].split(","):
            if l in LABELS:
                m = info[r["fname"]]
                rows.append({"fname": r["fname"], "split": split, "bucket": l, "licence": lic(m["license"]),
                             "uploader": m["uploader"], "title": m["title"], "tags": " ".join(m["tags"])})
out = ROOT / "manifests" / "fsd50k_classes.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
from collections import Counter
print(len(rows), "rows", dict(Counter(r["bucket"] for r in rows)))
