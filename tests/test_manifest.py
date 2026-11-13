"""The corpus manifest is consistent: counts match manifests/SUMMARY.md, only the six classes appear, and no recording group is in training and in a held-out split."""
from pathlib import Path

import pandas as pd

M = Path(__file__).resolve().parent.parent / "manifests"
CLASSES = {"chainsaw", "blast", "fire", "wood", "wind", "background"}


def load():
    return pd.read_csv(M / "items.csv")


def test_counts_match_the_summary():
    d = load(); assert len(d) == 43715
    assert d.status.value_counts().to_dict() == {"ok": 43055, "leak": 357, "dup": 198, "conflict": 103, "unreadable": 2}


def test_only_the_six_classes_and_known_licence_classes():
    ok = load().query("status == 'ok'")
    assert set(ok.cls) == CLASSES and set(ok.lic_class) <= {"open", "unknown", "nc"}


def test_no_recording_group_is_in_training_and_in_a_held_out_split():
    ok = load().query("status == 'ok'"); train = set(ok[ok.split == "train"].group.dropna())
    for split in ("val", "test_in", "test_ext"):
        assert not train & set(ok[ok.split == split].group.dropna()), split
