"""Download the Belize forest gunshot set (Mendeley x48cwz364j v3, CC BY 4.0, 8 kHz, about 4 s clips).

Folder names are not exposed by the API, so folders are identified by file count (assumption, checked
afterwards with 03_belize_sanity.py): 597 = gunshot train, 150 = gunshot validation, 7040 = background
validation; the single 'background.zip' (1.3 GB) is the background training set.
Every file is checked against the sha256 in the listing; existing verified files are skipped.
Run: python3 -I scripts/02_belize_download.py
"""
import hashlib, json, time, urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "belize"
API = "https://data.mendeley.com/public-api/datasets/x48cwz364j"
HDR = {"Accept": "application/vnd.mendeley-public-dataset.1+json", "User-Agent": "Mozilla/5.0"}
NAMES = {597: "gunshot_train", 150: "gunshot_val", 7040: "background_val", 1: "zip"}


def get(url, tries=6):
    for i in range(tries):  # the server resets connections under load: back off and retry
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=HDR), timeout=120).read()
        except OSError:
            if i == tries - 1: raise
            time.sleep(2 ** i)


files = json.loads(get(API))["files"]
sizes = Counter(f["folder_id"] for f in files)
unknown = {n for n in sizes.values()} - set(NAMES)
assert not unknown, f"unexpected folder sizes {unknown}; re-check the folder mapping"
(OUT).mkdir(parents=True, exist_ok=True)


def one(f):
    cd = f["content_details"]
    dest = OUT / NAMES[sizes[f["folder_id"]]] / f["filename"]
    dest.parent.mkdir(exist_ok=True)
    if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() == cd["sha256_hash"]:
        return "skipped"
    data = get(cd["download_url"])
    if hashlib.sha256(data).hexdigest() != cd["sha256_hash"]:
        return "BAD_HASH " + f["filename"]
    dest.write_bytes(data)
    return "ok"


with ThreadPoolExecutor(4) as ex:
    res = Counter(r if r in ("ok", "skipped") else "bad" for r in ex.map(one, files))
print(dict(res), "of", len(files))
