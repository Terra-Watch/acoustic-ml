"""Fetch chosen FSD50K clips without downloading the 25 GB archives (HTTP range requests into the split zips).

Usage: python3 -I scripts/03_fsd50k_fetch.py MANIFEST.csv bucket1,bucket2 [subfolder]
(subfolder under data/fsd50k/, default "clips")
Reads the zip central directory once per split, then pulls only the listed clips into data/fsd50k/clips/.
Each clip is checked against the CRC32 in the archive. Zenodo record 4060432.
"""
import csv, http.client, struct, sys, time, urllib.request, zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "fsd50k" / (sys.argv[3] if len(sys.argv) > 3 else "clips")
BASE = "https://zenodo.org/records/4060432/files/"
PARTS = {"dev": [f"FSD50K.dev_audio.z0{i}" for i in range(1, 6)] + ["FSD50K.dev_audio.zip"],
         "eval": ["FSD50K.eval_audio.z01", "FSD50K.eval_audio.zip"]}


def open_url(req, tries=6):
    for i in range(tries):  # transient resets, cut-off bodies, one-off certificate errors: retry, verification stays on
        try:
            return urllib.request.urlopen(req, timeout=120)
        except OSError:
            if i == tries - 1: raise
            time.sleep(2 ** i)


def rng(name, start, length, chunk=1 << 20):
    out = b""
    while len(out) < length:
        a, n = start + len(out), min(chunk, length - len(out))
        req = urllib.request.Request(f"{BASE}{name}?download=1", headers={"Range": f"bytes={a}-{a + n - 1}"})
        for i in range(6):
            try:
                piece = open_url(req).read()
                if len(piece) == n: break
            except (OSError, http.client.HTTPException):
                pass
            time.sleep(2 ** i)
        else:
            raise RuntimeError(f"could not read {name} bytes {a}-{a + n - 1}")
        out += piece
    return out


def part_sizes(names):
    sizes = []
    for n in names:
        r = open_url(urllib.request.Request(f"{BASE}{n}?download=1", method="HEAD"))
        sizes.append(int(r.headers["Content-Length"]))
    return sizes


def central_directory(names, sizes):
    last, size = names[-1], sizes[-1]
    tail = rng(last, size - 98, 98)
    z64 = tail.rfind(b"PK\x06\x06")
    assert z64 >= 0, "no zip64 end record"
    n_entries, cd_size, cd_off = struct.unpack("<QQQ", tail[z64 + 32:z64 + 56])
    cd = rng(last, cd_off, cd_size)
    out, p = {}, 0
    for _ in range(n_entries):
        assert cd[p:p + 4] == b"PK\x01\x02"
        method, = struct.unpack("<H", cd[p + 10:p + 12])
        crc, csize, usize, nlen, elen, clen = struct.unpack("<IIIHHH", cd[p + 16:p + 34])
        disk, = struct.unpack("<H", cd[p + 34:p + 36])
        off, = struct.unpack("<I", cd[p + 42:p + 46])
        assert 0xFFFFFFFF not in (csize, usize, off), "zip64 entry fields not handled"
        out[cd[p + 46:p + 46 + nlen].decode()] = (method, crc, csize, disk, off)
        p += 46 + nlen + elen + clen
    return out


def read_span(names, sizes, disk, off, length):
    buf = b""
    while length:
        take = min(length, sizes[disk] - off)
        buf += rng(names[disk], off, take)
        length -= take; disk += 1; off = 0
    return buf


def fetch(split, fnames, workers=6):
    names = PARTS[split]; sizes = part_sizes(names)
    cd = central_directory(names, sizes)
    key = {Path(k).stem: k for k in cd}

    def one(f):
        dest = OUT / f"{f}.wav"
        if dest.exists(): return 0
        method, crc, csize, disk, off = cd[key[f]]
        hdr = read_span(names, sizes, disk, off, 30)
        nlen, elen = struct.unpack("<HH", hdr[26:30])
        d0, o0 = disk, off + 30 + nlen + elen      # data starts after the local header and can run into the next part
        while o0 >= sizes[d0]: o0 -= sizes[d0]; d0 += 1
        raw = read_span(names, sizes, d0, o0, csize)
        data = raw if method == 0 else zlib.decompressobj(-15).decompress(raw)
        assert zlib.crc32(data) == crc, f"CRC mismatch {f}"
        tmp = dest.with_suffix(".part"); tmp.write_bytes(data); tmp.replace(dest)   # never leave half a file under the final name
        return 1

    done = failed = 0
    with ThreadPoolExecutor(workers) as ex:
        futures = {ex.submit(one, f): f for f in fnames}
        for i, fut in enumerate(as_completed(futures), 1):
            try: done += fut.result()
            except Exception as e: failed += 1; print("FAILED", futures[fut], repr(e)[:80], flush=True)
            if i % 50 == 0: print(f"{split}: {i}/{len(fnames)} checked, {done} fetched, {failed} failed", flush=True)
    print(f"{split}: {done} fetched, {failed} failed of {len(fnames)}", flush=True)


if __name__ == "__main__":
    manifest, buckets = sys.argv[1], set(sys.argv[2].split(","))
    OUT.mkdir(parents=True, exist_ok=True)
    want = {"dev": [], "eval": []}
    for r in csv.DictReader(open(manifest)):
        if r["bucket"] in buckets: want[r["split"]].append(r["fname"])
    for split, fn in want.items():
        if fn: fetch(split, fn)
    print("done ->", OUT, len(list(OUT.glob("*.wav"))), "files")
