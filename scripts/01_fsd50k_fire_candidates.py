"""List FSD50K clips labelled Fire, with per-clip licence and an outdoor/household guess.

Downloads only the metadata (about 7 MB) from Zenodo record 4060432; no audio.
The outdoor/household bucket is a keyword guess on title, tags and description. It is a shortlist
for listening, not a label.
Run: python3 -I scripts/01_fsd50k_fire_candidates.py
"""
import csv, json, re, urllib.request, zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA, OUT = ROOT / "data" / "fsd50k", ROOT / "manifests" / "fsd50k_fire_candidates.csv"
BASE = "https://zenodo.org/records/4060432/files/"
LABELS = {"Fire"}  # Crackle-only clips are vinyl, plastic and static: not used
OUTDOOR = re.compile(r"camp ?fire|bon ?fire|wild ?fire|forest|brush|bushfire|burning (wood|leaves|leaf|grass|brush|tree|log|trash|garbage|straw|residue)|open fire|fire pit|woodland|outdoor")
HOUSEHOLD = re.compile(r"lighter|stove|fire ?place|match(es)?\b|candle|gas |oven|cigarette|fireworks?|sparkler|toaster|grill|bbq|barbecue|heater|kettle|frying|sizzl")


def fetch(name, members):
    DATA.mkdir(parents=True, exist_ok=True)
    zpath = DATA / name
    if not zpath.exists():
        urllib.request.urlretrieve(f"{BASE}{name}?download=1", zpath)
    with zipfile.ZipFile(zpath) as z:
        for m in z.namelist():
            if Path(m).name in members:  # extract only the files we need, flat
                (DATA / Path(m).name).write_bytes(z.read(m))


def licence(url):
    url = url or ""
    if "zero" in url or "publicdomain" in url: return "CC0"
    if "by-nc" in url: return "BY-NC"
    if "sampling" in url: return "Sampling+"
    return "BY" if "/by/" in url else url


def bucket(text):
    if HOUSEHOLD.search(text): return "household"
    return "outdoor" if OUTDOOR.search(text) else "unknown"


fetch("FSD50K.ground_truth.zip", {"dev.csv", "eval.csv"})
fetch("FSD50K.metadata.zip", {"dev_clips_info_FSD50K.json", "eval_clips_info_FSD50K.json"})
info = {}
for f in ("dev_clips_info_FSD50K.json", "eval_clips_info_FSD50K.json"):
    info.update(json.loads((DATA / f).read_text()))

rows = []
for split in ("dev", "eval"):
    for r in csv.DictReader(open(DATA / f"{split}.csv")):
        got = LABELS & set(r["labels"].split(","))
        if not got: continue
        m = info[r["fname"]]
        text = f'{m["title"]} {" ".join(m["tags"])} {m["description"]}'.lower()
        rows.append({"fname": r["fname"], "split": split, "label": "+".join(sorted(got)), "licence": licence(m["license"]),
                     "bucket": bucket(text), "uploader": m["uploader"], "title": m["title"], "tags": " ".join(m["tags"])})

OUT.parent.mkdir(exist_ok=True)
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(len(rows), "clips ->", OUT)
print(Counter((r["bucket"], "open" if r["licence"] in ("CC0", "BY") else "restricted") for r in rows))
