"""Pack the usable items into one small corpus the cloud VMs can train from: mono FLAC, sample rate at most 16 kHz (native rate kept below that), every item cut to its interval and
capped in length. Belize background is thinned (it is 35,000 near-identical clips) in the training splits only; every held-out clip is kept.
Output: data/corpus_v1/<split>/<id>.flac and data/corpus_v1/corpus.csv. Run: .../ml/venv/bin/python -I scripts/11_pack_corpus.py
"""
import csv, hashlib, sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

HERE = Path(__file__).resolve().parent.parent
ROOTS = {"patrol": Path("/Users/kireeti/Desktop/Projects/SECURE-FOREST-PATROL/ml/datasets/raw"), "acoustic": HERE / "data"}
OUT = HERE / "data" / "corpus_v1"
MAX_S = {"sensing_forest": 120.0}; DEFAULT_MAX_S = 60.0
BELIZE_BG_TRAIN_CAP = 6000


def pack(job):
    i, r = job
    src = ROOTS[r["root"]] / r["path"]
    dest = OUT / r["split"] / f"{i}.flac"
    if dest.exists(): return i, str(dest.relative_to(OUT)), None
    try:
        info = sf.info(str(src)); sr = info.samplerate
        start = float(r["start"]); end = float(r["end"]) if r["end"] != "" else info.frames / sr
        cap = MAX_S.get(r["source"], DEFAULT_MAX_S)
        if end - start > cap:                                    # keep the middle part of a long recording
            mid = (start + end) / 2; start, end = mid - cap / 2, mid + cap / 2
        x, _ = sf.read(str(src), start=int(start * sr), stop=int(end * sr), dtype="float32", always_2d=True)
        x = x.mean(axis=1)
        if sr > 16000:
            g = np.gcd(sr, 16000); x = resample_poly(x, 16000 // g, sr // g).astype("float32"); sr = 16000
        dest.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(dest), x, sr, format="FLAC", subtype="PCM_16")
        return i, str(dest.relative_to(OUT)), (sr, round(len(x) / sr, 2))
    except Exception as e:
        return i, "", repr(e)[:80]


if __name__ == "__main__":
    rows = [r for r in csv.DictReader(open(HERE / "manifests" / "items.csv")) if r["status"] == "ok"]
    keep = []
    bz = [r for r in rows if r["source"] == "belize" and r["cls"] == "background" and r["split"] != "test_ext"]
    bz_keep = {id(r) for r in sorted(bz, key=lambda r: hashlib.sha1((r["path"]).encode()).hexdigest())[:BELIZE_BG_TRAIN_CAP]}
    for r in rows:
        if r in bz and id(r) not in bz_keep: continue
        keep.append(r)
    jobs = [(f"{r['source']}_{hashlib.sha1('|'.join((r['root'], r['path'], str(r['start']), str(r['end']))).encode()).hexdigest()[:12]}", r) for r in keep]   # stable id: does not change when items are added
    with Pool(8) as p: res = p.map(pack, jobs, chunksize=64)
    bad = 0; out_rows = []
    for (i, r), (_, rel, meta) in zip(jobs, res):
        if not rel or isinstance(meta, str): bad += 1; continue
        sr, dur = meta if meta else (sf.info(str(OUT / rel)).samplerate, round(sf.info(str(OUT / rel)).duration, 2))
        out_rows.append(dict(id=i, path=rel, cls=r["cls"], sub=r["sub"], source=r["source"], group=r["group"], split=r["split"], cv_fold=r["cv_fold"], sr=sr, dur=dur, lic_class=r["lic_class"],
                             rule=r["rule"], note=r["note"]))
    with open(OUT / "corpus.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0])); w.writeheader(); w.writerows(out_rows)
    from collections import Counter
    print(len(out_rows), "items packed,", bad, "failed;", dict(Counter(r["cls"] for r in out_rows)))
