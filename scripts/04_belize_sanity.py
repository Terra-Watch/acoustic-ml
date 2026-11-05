"""Check the folder mapping assumed by 02_belize_download.py: gunshot folders should be short loud transients
(high crest factor), background folders should not. Also checks rate and duration.
Run: ~/Desktop/Projects/SECURE-FOREST-PATROL/ml/venv/bin/python -I scripts/04_belize_sanity.py
"""
import wave
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parent.parent / "data" / "belize"
for d in sorted(p for p in root.iterdir() if p.is_dir()):
    crest, peak, bad = [], [], 0
    files = sorted(d.glob("*.WAV")) + sorted(d.glob("*.wav"))
    for f in files:
        with wave.open(str(f)) as w:
            if w.getframerate() != 8000 or w.getnchannels() != 1 or abs(w.getnframes() / 8000 - 4.0) > 0.1: bad += 1
            x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(float) / 32768
        rms = np.sqrt(np.mean(x ** 2)) + 1e-9
        crest.append(np.abs(x).max() / rms); peak.append(np.abs(x).max())
    print(f"{d.name}: {len(files)} files, {bad} off-format, median crest {np.median(crest):.1f}, median peak {np.median(peak):.3f}")
