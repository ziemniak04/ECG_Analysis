"""Which MIT-BIH records the beats of each class come from, per split.

Run:  python -m src.ml.split_report
Useful to see how many patients are behind the rare classes (S and F).
"""
from collections import Counter

import numpy as np

from src.ml.data import MITBIH_CLASSES, PARTS, load_mitbih, mitbih_group_split


def main():
    d = load_mitbih()
    split = mitbih_group_split(d["y"], d["groups"])

    for part in PARTS:
        idx = split[part]
        rec = d["record_ids"][idx]
        y = d["y"][idx]
        print(f"\n== {part}: {len(np.unique(rec))} records ==")
        print("records:", " ".join(sorted(np.unique(rec).tolist())))
        for k, name in enumerate(MITBIH_CLASSES):
            counts = Counter(rec[y == k].tolist()).most_common(6)
            text = "  ".join(f"{r}:{n}" for r, n in counts)
            print(f"  {name} ({int((y == k).sum())} beats): {text}")


if __name__ == "__main__":
    main()