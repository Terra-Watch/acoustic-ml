"""Fetch Freesound sounds by id: the HQ preview (MP3, about 128 kbps) plus licence and attribution into a manifest.

The API key is read from website/backend/.env (FREESOUND_API_KEY). Originals need an OAuth2 login, so previews
are used; they are lossy. Only the sound ids you pass are downloaded.
Run: python3 -I scripts/05_freesound_fetch.py 253770 752028 519520
"""
import csv, json, sys, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV = ROOT.parent / "website" / "backend" / ".env"
OUT = ROOT / "data" / "freesound"
MANIFEST = ROOT / "manifests" / "freesound_downloads.csv"
FIELDS = "id,name,username,license,duration,samplerate,type,url,previews,tags,description"
key = dict(l.strip().split("=", 1) for l in ENV.read_text().splitlines() if l.startswith("FREESOUND_"))["FREESOUND_API_KEY"]
auth = {"Authorization": "Token " + key}


def get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=auth), timeout=120).read()


OUT.mkdir(parents=True, exist_ok=True)
rows = []
for sid in sys.argv[1:]:
    m = json.loads(get(f"https://freesound.org/apiv2/sounds/{sid}/?fields={FIELDS}"))
    dest = OUT / f"{sid}.mp3"
    if not dest.exists():
        dest.write_bytes(get(m["previews"]["preview-hq-mp3"]))
    rows.append({"id": sid, "file": dest.name, "name": m["name"], "username": m["username"], "license": m["license"],
                 "duration_s": round(m["duration"], 1), "orig_samplerate": m["samplerate"], "orig_type": m["type"],
                 "url": m["url"], "tags": " ".join(m["tags"][:12])})
    print(sid, m["license"].rstrip("/").split("/")[-2:], round(dest.stat().st_size / 1e6, 1), "MB")
old = list(csv.DictReader(open(MANIFEST))) if MANIFEST.exists() else []
rows = [r for r in old if r["id"] not in {x["id"] for x in rows}] + rows
with open(MANIFEST, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
