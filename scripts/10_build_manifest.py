"""Build the one manifest every later step reads: items.csv (every audio item with class, source, group, split, licence, status) and SUMMARY.md.

Item = a whole clip, or an interval of a long recording. Classes: chainsaw, blast (gunshot + fireworks), fire, wood, wind, background.
Splits (fixed here, never changed later):
  test_ext  whole sources or periods the model never trains on: Belize validation period, the held-out C3GD site, two Rodopi recordings, Sensing the Forest, real wildfire
  train / val / test_in  assigned by hash of `group` (80/10/10) for everything else, so clips of one recording or one Freesound upload never straddle splits
  cv_fold (0-4) from the same hash, for grouped cross-validation
Freesound ids appear in FSD50K, ESC-50 and FSC22: an id is kept once (priority FSD50K > ESC-50 > FSC22); the others get status 'dup'.
Run: /Users/kireeti/Desktop/Projects/SECURE-FOREST-PATROL/ml/venv/bin/python -I scripts/10_build_manifest.py
"""
import csv, hashlib, json, re, sys
from collections import Counter, defaultdict
from pathlib import Path

import soundfile as sf

HERE = Path(__file__).resolve().parent.parent
PATROL = Path("/Users/kireeti/Desktop/Projects/SECURE-FOREST-PATROL/ml")
ROOTS = {"patrol": PATROL / "datasets" / "raw", "acoustic": HERE / "data"}
sys.path.insert(0, str(PATROL / "scripts"))
import textgrid  # noqa: E402  (the patrol project's Praat TextGrid parser)

rows, hold = [], []


def add(root, rel, source, cls, sub, group, licence, lic_class, start=0.0, end=None, rule="any", split=None, status="ok", note="", fsid=""):
    rows.append(dict(root=root, path=str(rel), source=source, cls=cls, sub=sub, group=group, licence=licence, lic_class=lic_class, start=start, end=end if end is not None else "",
                     rule=rule, split=split or "", status=status, note=note, fsid=fsid))


def licclass(lic):
    l = (lic or "").lower()
    if "nc" in l: return "nc"
    if l in ("cc0", "cc by 4.0", "by", "cc by", "public domain") or l.startswith("cc by 4") or "zero" in l: return "open"
    return "unknown"


# ---- Belize (CC BY 4.0). Folder meaning was checked in 04_belize_sanity.py. File name starts with the UNIX time in hex: group = UTC hour (one recording session)
import datetime
def belize_day(name):   # (name kept; it now returns the hour)
    try: return datetime.datetime.fromtimestamp(int(name[:8], 16), datetime.timezone.utc).strftime("%Y%m%dT%H")
    except ValueError: return "unknown"
for folder, cls, sub, split in (("gunshot_train", "blast", "gunshot", None), ("gunshot_val", "blast", "gunshot", "test_ext"),
                                ("background_train", "background", "forest_survey", None), ("background_val", "background", "forest_survey", "test_ext")):
    for p in sorted((HERE / "data" / "belize" / folder).glob("*.[Ww][Aa][Vv]")):
        add("acoustic", p.relative_to(ROOTS["acoustic"]), "belize", cls, sub, f"belize:{belize_day(p.name)}", "CC BY 4.0", "open", rule="peak" if cls == "blast" else "any", split=split,
            note="8 kHz; background includes false positives of the authors' gunshot filter" if cls == "background" else "8 kHz")

# ---- C3GD (CC BY 4.0): all 12k clips are on disk; cap per site x firearm x cartridge so no combination dominates. The ny_gravel_pit site is held out whole.
c3 = PATROL / "datasets" / "raw" / "c3gd" / "C3GD"
per, CAP = Counter(), 12
for r in sorted(csv.DictReader(open(c3 / "metadata.csv")), key=lambda r: r["filename"]):
    combo = (r["event_id"], r["platform_id"], r["cartridge_id"])
    if per[combo] >= CAP: continue
    per[combo] += 1
    add("patrol", Path("c3gd/C3GD/data") / r["filename"], "c3gd", "blast", "gunshot", f"c3gd:{r['event_id']}:{r['platform_id']}:{r['day']}", "CC BY 4.0", "open", rule="peak",
        split="test_ext" if r["event_id"] == "ny_gravel_pit" else None, note=f"{r['event_id']} {r['platform_id']}")

# ---- FSC22 (Freesound clips, licences differ per clip and were not recorded: 'unknown'). Voices are left out (privacy).
FSC = {"Chainsaw": ("chainsaw", ""), "Gunshot": ("blast", "gunshot"), "Firework": ("blast", "firework"), "Fire": ("fire", "campfire"),
       "TreeFalling": ("wood", "tree_falling"), "Axe": ("wood", "axe"), "WoodChop": ("wood", "wood_chop"), "Handsaw": ("wood", "handsaw"), "Wind": ("wind", ""),
       **{k: ("background", k.lower()) for k in ("Rain", "Thunderstorm", "WaterDrops", "Insect", "Frog", "BirdChirping", "WingFlaping", "Lion", "WolfHowl", "Squirrel", "Silence",
                                                  "Footsteps", "Generator", "VehicleEngine", "Helicopter")}}
for r in csv.DictReader(open(PATROL / "datasets/raw/fsc22/FSC22_repo/Metadata/Metadata V1.0 FSC22.csv")):
    m = FSC.get(r["Class Name"])
    if not m: continue
    fsid = re.match(r"(\d+)", r["Source File Name"]); fsid = fsid.group(1) if fsid else ""
    add("patrol", Path("fsc22/FSC22_repo/Audios") / r["Dataset File Name"], "fsc22", m[0], m[1] or r["Class Name"].lower(), f"fs:{fsid}" if fsid else f"fsc22:{r['Source File Name']}", "unknown (Freesound, per clip)", "unknown",
        rule="peak" if m[0] == "blast" else "any", fsid=fsid)

# ---- ESC-50 (CC BY-NC)
ESC = {"chainsaw": ("chainsaw", ""), "fireworks": ("blast", "firework"), "crackling_fire": ("fire", "campfire"), "hand_saw": ("wood", "handsaw"), "door_wood_creaks": ("wood", "wood_creak"), "wind": ("wind", ""),
       **{k: ("background", k) for k in ("rain", "chirping_birds", "crickets", "crow", "frog", "insects", "thunderstorm", "water_drops", "footsteps", "engine", "helicopter")}}
esc_files = {p.name: p for p in (PATROL / "datasets/raw/esc50_hf/ESC-50-master").rglob("*.wav")}   # the HF mirror nests clips inside per-category folders; only some categories were downloaded
for r in csv.DictReader(open(PATROL / "datasets/raw/esc50_hf/ESC-50-master/meta/esc50.csv")):
    m = ESC.get(r["category"])
    if m and r["filename"] in esc_files: add("patrol", esc_files[r["filename"]].relative_to(ROOTS["patrol"]), "esc50", m[0], m[1] or r["category"], f"fs:{r['src_file']}", "CC BY-NC 3.0", "nc", rule="peak" if m[0] == "blast" else "any", fsid=r["src_file"])

# ---- FSD50K (per-clip licence from its metadata). Fire clips are the 18 you confirmed by ear; the 115 unclear ones and 'Sawing' wait for listening.
meta = {r["fname"]: r for r in csv.DictReader(open(HERE / "manifests" / "fsd50k_fire_candidates.csv"))}
for r in csv.DictReader(open(HERE / "manifests" / "fsd50k_fire_listening.csv")):
    if r["verdict (open_fire / household / other / unusable)"] == "open_fire":
        fid = r["file"][:-4]; m = meta[fid]
        add("acoustic", Path("fsd50k/clips") / r["file"], "fsd50k", "fire", "open_fire", f"fs:{fid}", m["licence"], licclass(m["licence"]), fsid=fid, note="listened: open fire")
for r in csv.DictReader(open(HERE / "manifests" / "fsd50k_fire_candidates.csv")):
    p = ROOTS["acoustic"] / "fsd50k/fire_unclear" / f"{r['fname']}.wav"
    if p.exists(): add("acoustic", p.relative_to(ROOTS["acoustic"]), "fsd50k", "fire", "open_fire", f"fs:{r['fname']}", r["licence"], licclass(r["licence"]), fsid=r["fname"], note="listened to and confirmed by the owner (2026-11-02)")
FSDC = {"Wood": ("wood", "wood"), "Wind": ("wind", ""), "Rain": ("background", "rain"), "Gunshot_and_gunfire": ("blast", "gunshot"), "Fireworks": ("blast", "firework")}
for r in csv.DictReader(open(HERE / "manifests" / "fsd50k_classes.csv")):
    p = ROOTS["acoustic"] / "fsd50k/classes" / f"{r['fname']}.wav"
    if r["bucket"] == "Sawing":
        if p.exists(): add("acoustic", p.relative_to(ROOTS["acoustic"]), "fsd50k", "wood", "sawing", f"fs:{r['fname']}", r["licence"], licclass(r["licence"]), fsid=r["fname"], note="listened to and confirmed by the owner (2026-11-02); label wood, not chainsaw")
        continue
    c, sub = FSDC[r["bucket"]]
    if p.exists(): add("acoustic", p.relative_to(ROOTS["acoustic"]), "fsd50k", c, sub or r["bucket"].lower(), f"fs:{r['fname']}", r["licence"], licclass(r["licence"]), rule="peak" if c == "blast" else "any", fsid=r["fname"],
                       note="FSD50K label only, not yet listened to")

# ---- Freesound clips collected by search term for the thin classes (script 12): CC0 or CC BY, label = the search term only, NOT yet listened to
tf = HERE / "manifests" / "freesound_targeted.csv"
if tf.exists():
    for r in csv.DictReader(open(tf)):
        lic = "CC0" if "zero" in r["license"] or "publicdomain" in r["license"] else "CC BY 4.0"
        add("acoustic", Path("freesound_targeted") / r["file"], "freesound_targeted", r["cls"], r["term"].replace(" ", "_"), f"fs:{r['id']}", lic, "open", fsid=str(r["id"]),
            note=f"unreviewed; found by search '{r['term']}' ({r['username']})")

# ---- Rodopi (Greek forest, 8 kHz): intervals from the TextGrids; two recordings with chainsaw held out whole
rod = PATROL / "datasets" / "raw" / "rodopi"
tg = sorted(rod.glob("*.TextGrid"))
with_saw = [p for p in tg if any(i.label.strip().lower() == "saw" for i in textgrid.parse_textgrid(p))]
held = {p.stem for p in with_saw[-2:]}
for p in tg:
    for i in textgrid.parse_textgrid(p):
        lab = i.label.strip().lower()
        if not lab or i.xmax - i.xmin < 1.0: continue
        c = "chainsaw" if lab == "saw" else "background"
        add("patrol", Path("rodopi") / f"{p.stem}.wav", "rodopi", c, "saw" if c == "chainsaw" else lab, f"rodopi:{p.stem}", "unknown (Zenodo 5824433)", "unknown", start=i.xmin, end=i.xmax,
            split="test_ext" if p.stem in held else None, note="8 kHz recorder, interval from TextGrid")

# ---- FrugalAI (CC BY-NC; clips cannot be grouped by recording: one group per file)
for cls, folder in (("chainsaw", "chainsaw"), ("background", "environment")):
    for p in sorted((PATROL / "datasets/raw/rfcx_frugalai/audio" / folder).glob("*.wav")):
        add("patrol", p.relative_to(ROOTS["patrol"]), "frugalai", cls, folder, f"frugalai:{p.stem}", "CC BY-NC 4.0", "nc", note="clips of one recording cannot be told apart")

# ---- Sensing the Forest (CC0, UK): kept whole as the external set for false alarms per hour
for p in sorted((PATROL / "datasets/raw/sensing_forest/audio").glob("*.wav")):
    add("patrol", p.relative_to(ROOTS["patrol"]), "sensing_forest", "background", "ambient", f"stf:{p.stem}", "CC0 1.0", "open", split="test_ext")

# ---- Real wildfire recordings (Freesound previews): test only
for r in csv.DictReader(open(HERE / "manifests" / "freesound_downloads.csv")):
    lic = r["license"].rstrip("/").split("/")[-2:]; l = "CC0" if "zero" in lic[0] or "publicdomain" in r["license"] else "CC BY-NC 4.0"
    add("acoustic", Path("freesound") / r["file"], "freesound_real_fire", "fire", "wildfire", f"fs:{r['id']}", l, licclass(l), split="test_ext", note="lossy MP3 preview; real wildfire; test only")

# ---- cross-dataset duplicates by Freesound id
prio = {"fsd50k": 0, "esc50": 1, "fsc22": 2}
by_id = defaultdict(list)
for i, r in enumerate(rows):
    if r["fsid"] and r["status"] == "ok": by_id[r["fsid"]].append(i)
dups = conflicts = 0
WHOLE = {"fsd50k", "freesound_targeted"}      # one file per upload; ESC-50 and FSC22 cut several 5 s segments from one upload, so two classes under one id there are different segments, not a conflict
AMBIENT = {"wind", "background"}


def resolve(classes, subs):
    """Which class wins when sources disagree about one Freesound upload, or None to exclude it. Rules: a specific label beats a general one (chainsaw over FSD50K 'Sawing');
    an open fire beats weather and wood noise; a wood event beats wind; everything else (ambient vs ambient, alert vs alert, alert vs ambient) is too uncertain and is excluded."""
    if classes == {"chainsaw", "wood"} and "sawing" in subs: return "chainsaw"
    if "fire" in classes and classes <= {"fire", "wind", "wood", "background"}: return "fire"
    if "wood" in classes and classes <= {"wood", "wind", "background"}: return "wood"
    return None


log = []
for ids in by_id.values():
    if len(ids) < 2: continue
    clash = {i for i in ids for j in ids if i < j and rows[i]["cls"] != rows[j]["cls"] and (rows[i]["source"] != rows[j]["source"] or rows[i]["source"] in WHOLE)}
    clash |= {j for i in ids for j in ids if i < j and rows[i]["cls"] != rows[j]["cls"] and (rows[i]["source"] != rows[j]["source"] or rows[i]["source"] in WHOLE)}
    if clash:
        w = resolve({rows[i]["cls"] for i in clash}, {rows[i]["sub"] for i in clash})
        for i in clash:
            if rows[i]["cls"] != w:
                rows[i]["status"] = "conflict"; conflicts += 1
                rows[i]["note"] += f"; excluded: same Freesound id labelled {sorted({rows[k]['cls'] for k in clash})}" + (f", resolved to {w}" if w else ", unresolved")
            log.append((rows[i]["fsid"], rows[i]["source"], rows[i]["cls"], rows[i]["sub"], w or "excluded"))
    live = [i for i in ids if rows[i]["status"] == "ok"]
    live.sort(key=lambda i: prio.get(rows[i]["source"], 9))
    if live:
        keep = rows[live[0]]
        for i in live[1:]:
            if rows[i]["source"] == keep["source"] or rows[i]["cls"] != keep["cls"]: continue    # same dataset: kept (one group); different class here means different segments
            rows[i]["status"] = "dup"; dups += 1
seen = {}                                                          # the same file and interval listed twice with the same class
for i, r in enumerate(rows):
    if r["status"] != "ok": continue
    k = (r["root"], r["path"], r["start"], r["end"])
    if k not in seen: seen[k] = i; continue
    r["status"] = "dup"; dups += 1
with open(HERE / "manifests" / "label_conflicts.csv", "w", newline="") as f:
    w_ = csv.writer(f); w_.writerow(["freesound_id", "source", "class", "sub", "outcome"]); w_.writerows(sorted(set(log)))

# ---- splits by hash of group
def u(group): return int(hashlib.sha1(group.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
def fold(group): return int(hashlib.sha1(group.encode()).hexdigest()[8:12], 16) % 5
for r in rows:
    if not r["split"]:
        x = u(r["group"]); r["split"] = "train" if x < 0.8 else "val" if x < 0.9 else "test_in"
    r["cv_fold"] = fold(r["group"])

# ---- leakage guard: any group that has an item in test_ext must not appear in train, val or test_in (the Belize validation period shares days with its training period;
# FSC22 cut segments out of one of the real wildfire recordings). Those rows are excluded, not moved.
ext_groups = {r["group"] for r in rows if r["split"] == "test_ext" and r["status"] == "ok"}
for r in rows:
    if r["split"] != "test_ext" and r["status"] == "ok" and r["group"] in ext_groups:
        r["status"] = "leak"; r["note"] += "; excluded: same group as a held-out (test_ext) item"

# ---- header check of every file (rate, length): flags unreadable files
cache = {}
for r in rows:
    f = ROOTS[r["root"]] / r["path"]
    if f not in cache:
        try: i = sf.info(str(f)); cache[f] = (i.samplerate, i.frames / i.samplerate if i.samplerate else 0, i.channels)
        except Exception: cache[f] = None
    if cache[f] is None: r["status"] = "unreadable"; r["sr"] = r["file_dur"] = r["ch"] = ""
    else: r["sr"], r["file_dur"], r["ch"] = cache[f][0], round(cache[f][1], 2), cache[f][2]
    s, e = float(r["start"]), float(r["end"]) if r["end"] != "" else None
    r["dur"] = round((e if e is not None else (cache[f][1] if cache[f] else 0)) - s, 2)

cols = ["root", "path", "source", "cls", "sub", "group", "split", "cv_fold", "start", "end", "dur", "sr", "ch", "rule", "licence", "lic_class", "status", "fsid", "note"]
out = HERE / "manifests" / "items.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)

ok = [r for r in rows if r["status"] == "ok"]
mins = lambda rs: round(sum(float(r["dur"]) for r in rs) / 60, 1)
lines = ["# Manifest summary (generated by scripts/10_build_manifest.py)", "", f"{len(rows)} items; status: {dict(Counter(r['status'] for r in rows))}; Freesound-id duplicates dropped: {dups} (class conflicts {conflicts})", "",
         "## Minutes of audio per class and split (status ok)", "", "| class | train | val | test_in | test_ext | items |", "|---|---|---|---|---|---|"]
for c in ("chainsaw", "blast", "fire", "wood", "wind", "background"):
    rs = [r for r in ok if r["cls"] == c]
    lines.append(f"| {c} | " + " | ".join(str(mins([r for r in rs if r['split'] == s])) for s in ("train", "val", "test_in", "test_ext")) + f" | {len(rs)} |")
lines += ["", "## Items per class and source (status ok)", "", "| class | " + " | ".join(sorted({r['source'] for r in ok})) + " |", "|---|" + "---|" * len({r['source'] for r in ok})]
for c in ("chainsaw", "blast", "fire", "wood", "wind", "background"):
    lines.append(f"| {c} | " + " | ".join(str(sum(1 for r in ok if r['cls'] == c and r['source'] == s)) for s in sorted({r['source'] for r in ok})) + " |")
lines += ["", "## Licence class of usable items", "", str(dict(Counter((r['cls'], r['lic_class']) for r in ok)))]
lines += ["", "## Sample rates (status ok)", "", str(dict(Counter(r['sr'] for r in ok)))]
lines += ["", "## Not usable yet", "", f"hold (needs listening): {sum(1 for r in rows if r['status']=='hold')}; unreadable: {[r['path'] for r in rows if r['status']=='unreadable'][:5]}"]
(HERE / "manifests" / "SUMMARY.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
