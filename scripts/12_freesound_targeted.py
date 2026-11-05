"""Collect open-licence Freesound clips for the thin classes (chainsaw, fire, wood) with the API key in website/backend/.env.

Only CC0 and CC BY clips, 1 to 60 s long, ids not already present in items.csv (FSD50K, ESC-50 and FSC22 are Freesound too). HQ MP3 previews are downloaded (the originals need OAuth).
Labels come from the search term only: every clip is marked unreviewed and a listening sheet is written for a random sample. Resumable.
Run: python3 -I scripts/12_freesound_targeted.py
"""
import csv, json, random, time, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
env = dict(l.strip().split("=", 1) for l in (HERE.parent / "website/backend/.env").read_text().splitlines() if l.startswith("FREESOUND_"))
H = {"Authorization": "Token " + env["FREESOUND_API_KEY"]}
OUT = HERE / "data" / "freesound_targeted"; OUT.mkdir(parents=True, exist_ok=True)
PLAN = {"chainsaw": ("chainsaw", ["chainsaw", "chain saw", "chainsaw forest"], 300),
        "fire": ("fire", ["campfire", "bonfire", "fire crackling", "forest fire", "burning wood", "wildfire"], 360),
        "wood": ("wood", ["branch snap", "tree falling"], 200)}
LIC = 'license:("Creative Commons 0" OR "Attribution") duration:[1 TO 60]'
have = {r["fsid"] for r in csv.DictReader(open(HERE / "manifests" / "items.csv")) if r["fsid"]}


def get(url, tries=5):
    for i in range(tries):
        try: return urllib.request.urlopen(urllib.request.Request(url, headers=H), timeout=60).read()
        except Exception:
            if i == tries - 1: raise
            time.sleep(2 ** i)


cands = {}
for cls, (label, terms, cap) in PLAN.items():
    for term in terms:
        page = 1
        while True:
            q = urllib.parse.urlencode({"query": term, "filter": LIC, "fields": "id,name,license,duration,username,previews,tags", "page_size": 150, "page": page})
            r = json.loads(get("https://freesound.org/apiv2/search/text/?" + q))
            for s in r["results"]:
                if str(s["id"]) not in have and s["id"] not in cands: cands[s["id"]] = (label, term, s)
            if not r.get("next") or sum(1 for v in cands.values() if v[0] == label) >= cap * 1.3: break
            page += 1
rng = random.Random(20261102); chosen = []
for cls, (label, terms, cap) in PLAN.items():
    pool = sorted((v for v in cands.values() if v[0] == label), key=lambda v: v[2]["id"]); rng.shuffle(pool); chosen += pool[:cap]
print({c: sum(1 for v in chosen if v[0] == c) for c in PLAN}, flush=True)


def one(v):
    label, term, s = v; dest = OUT / f"{s['id']}.mp3"
    if not dest.exists():
        data = get(s["previews"]["preview-hq-mp3"]); tmp = dest.with_suffix(".part"); tmp.write_bytes(data); tmp.replace(dest)
    return dict(id=s["id"], file=dest.name, cls=label, term=term, name=s["name"], username=s["username"], license=s["license"], duration=round(s["duration"], 1), url=f"https://freesound.org/s/{s['id']}/", tags=" ".join(s["tags"][:10]))


rows, failed = [], 0
with ThreadPoolExecutor(6) as ex:
    futs = [ex.submit(one, v) for v in chosen]
    for i, f in enumerate(futs, 1):
        try: rows.append(f.result())
        except Exception as e: failed += 1
        if i % 50 == 0: print(i, len(rows), "downloaded,", failed, "failed", flush=True)
with open(HERE / "manifests" / "freesound_targeted.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
sample = []
for cls in PLAN:
    pool = [r for r in rows if r["cls"] == cls]; rng.shuffle(pool); sample += pool[:30]
with open(HERE / "manifests" / "freesound_targeted_listening.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["file", "cls", "name", "license", "duration", "verdict (ok / wrong / mixed)", "notes"], extrasaction="ignore"); w.writeheader()
    for r in sample: w.writerow({**r, "verdict (ok / wrong / mixed)": "", "notes": ""})
print("done", len(rows), "clips,", failed, "failed")
