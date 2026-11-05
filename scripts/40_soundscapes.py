"""Add real forest soundscapes as 'background' to the corpus: Western US collection (Sierra Nevada, 33 hour-long recordings, CC BY 4.0, Zenodo 7050014; Clapp et al., Cornell).
From each recording 20 non-overlapping one-minute clips at random times are cut, mono, 16 kHz FLAC, written to data/corpus_v1/<split>/ and appended to corpus.csv.
Split by recording day (the recordings of one day are not independent): 0509 and 0523 train, 0510 val, 0529 test_ext. All come from ONE site, so test_ext here is a new day, not a new place.
Run (on the VM, needs ~5 GB): python3 scripts/40_soundscapes.py   (a listening sheet of 40 random clips is written to manifests/soundscape_listening.csv)"""
import csv, hashlib, subprocess, sys, zipfile
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

HERE = Path(__file__).resolve().parent.parent
RAW = HERE / "data" / "soundscape_wus"; OUT = HERE / "data" / "corpus_v1"
URL = "https://zenodo.org/api/records/7050014/files/soundscape_data.zip/content"
CLIPS, CLIP_S, SR = 20, 60, 16000
SPLIT = {"20180509": "train", "20180523": "train", "20180510": "val", "20180529": "test_ext"}


def main():
    RAW.mkdir(parents=True, exist_ok=True); zp = RAW / "soundscape_data.zip"
    if not zp.exists():
        for _ in range(30):          # Zenodo drops long downloads: resume from the part file until curl reports success
            if subprocess.run(["curl", "-sL", "-C", "-", "--retry", "10", "--retry-all-errors", "--retry-delay", "5", "-o", str(zp) + ".part", URL]).returncode == 0: break
        else: raise SystemExit("soundscape download did not finish")
        (RAW / "soundscape_data.zip.part").rename(zp)
    import json, urllib.request
    rec = json.load(urllib.request.urlopen("https://zenodo.org/api/records/7050014"))
    want = next(f["checksum"] for f in rec["files"] if f["key"] == "soundscape_data.zip").split(":")[1]
    got = subprocess.run(["md5sum", str(zp)], capture_output=True, text=True).stdout.split()[0]
    assert got == want, f"checksum mismatch {got} != {want}"
    with zipfile.ZipFile(zp) as z: z.extractall(RAW)
    rows, sheet, skipped = [], [], 0; rng = np.random.default_rng(0)
    for f in sorted(RAW.rglob("*.flac")):
        day = f.stem.split("_")[2]; split = SPLIT.get(day)
        if split is None: continue
        try: info = sf.info(str(f))
        except sf.LibsndfileError: skipped += CLIPS; continue
        minutes = int(info.duration // CLIP_S)
        whole = None
        for m in sorted(rng.choice(minutes, min(CLIPS, minutes), replace=False)):
            lo, hi = int(m * CLIP_S * info.samplerate), int((m + 1) * CLIP_S * info.samplerate)
            try: x, sr = sf.read(str(f), start=lo, stop=hi, dtype="float32", always_2d=True)
            except sf.LibsndfileError:                 # seeking in some of these hour-long FLACs loses sync: read the file through once and slice it
                try:
                    if whole is None: whole = sf.read(str(f), dtype="float32", always_2d=True)[0]
                    x, sr = whole[lo:hi], info.samplerate
                except sf.LibsndfileError: skipped += 1; continue          # a damaged stretch: skip that clip
            x = x.mean(axis=1)
            if sr != SR: g = gcd(sr, SR); x = resample_poly(x, SR // g, sr // g).astype("float32")
            cid = "soundscape_wus_" + hashlib.sha1(f"{f.name}|{m}".encode()).hexdigest()[:12]; rel = f"{split}/{cid}.flac"
            (OUT / split).mkdir(parents=True, exist_ok=True); sf.write(str(OUT / rel), x, SR, format="FLAC", subtype="PCM_16")
            rows.append(dict(id=cid, path=rel, cls="background", sub="forest_soundscape", source="soundscape_wus", group=f"soundscape_wus:{day}", split=split,
                             cv_fold=int(hashlib.sha1(day.encode()).hexdigest()[8:12], 16) % 5, sr=SR, dur=CLIP_S, lic_class="open", rule="any", note="CC BY 4.0 Zenodo 7050014"))
    cols = list(next(csv.DictReader(open(OUT / "corpus.csv"))).keys())
    have = {r["id"] for r in csv.DictReader(open(OUT / "corpus.csv"))}
    with open(OUT / "corpus.csv", "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); [w.writerow(r) for r in rows if r["id"] not in have]
    pick = rng.choice(len(rows), 40, replace=False)
    (HERE / "manifests").mkdir(exist_ok=True)
    with open(HERE / "manifests" / "soundscape_listening.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["id", "path", "heard (background / wind / other)"]); [w.writerow([rows[i]["id"], rows[i]["path"], ""]) for i in pick]
    from collections import Counter
    print(skipped, "clips skipped (damaged audio);", len(rows), "clips,", dict(Counter(r["split"] for r in rows)), "minutes per split")


if __name__ == "__main__": main()
