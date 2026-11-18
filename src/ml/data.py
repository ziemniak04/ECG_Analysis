"""Loading the exported ECG data and making the train/val/test splits.

Run from the repo root:  python -m src.ml.data
"""
from collections import Counter
from pathlib import Path

import numpy as np

from src.prepare_dataset import load_record_shard
from src.ptbxl_rhythm import load_pilot_record

ROOT = Path(__file__).resolve().parents[2]
MITBIH_DIR = ROOT / "data/processed/ecg_dataset_v1"
PTBXL_DIR = ROOT / "data/processed/ptbxl_rhythm_v1"
CACHE_DIR = ROOT / "data/processed/cache"  # inside data/processed so git ignores it

MITBIH_CLASSES = ("N", "S", "V", "F", "Q")
PTBXL_CLASSES = ("SR", "AFIB", "STACH", "SBRAD")
PARTS = ("train", "val", "test")


def load_mitbih(use_cache=True):
    # X is (beats, 2 leads, 252 samples)
    # v2 cache also holds the R-peak positions (needed for the RR features)
    cache = CACHE_DIR / "mitbih_v2.npz"
    if use_cache and cache.is_file():
        with np.load(cache, allow_pickle=False) as f:
            return {k: f[k] for k in f.files}

    files = sorted((MITBIH_DIR / "recordings").glob("*.npz"))
    if len(files) != 48:
        raise FileNotFoundError(f"expected 48 MIT-BIH files, found {len(files)}")

    X, y, groups, rec, pos = [], [], [], [], []
    for path in files:
        shard = load_record_shard(path)  # also validates the file
        X.append(shard["X"])
        y.append(shard["y"])
        groups.append(shard["patient_groups"])
        rec.append(shard["record_ids"])
        pos.append(shard["annotation_positions"])

    data = {"X": np.concatenate(X), "y": np.concatenate(y),
            "groups": np.concatenate(groups), "record_ids": np.concatenate(rec),
            "positions": np.concatenate(pos)}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **data)
    return data


def load_ptbxl(use_cache=True):
    # X is (ecgs, 12 leads, 1000 samples), one file per ECG so this is the slow part
    cache = CACHE_DIR / "ptbxl_v1.npz"
    if use_cache and cache.is_file():
        with np.load(cache, allow_pickle=False) as f:
            return {k: f[k] for k in f.files}

    files = sorted((PTBXL_DIR / "recordings").glob("*_rhythm.npz"))
    if len(files) != 19605:
        raise FileNotFoundError(f"expected 19605 PTB-XL files, found {len(files)}")

    n = len(files)
    X = np.zeros((n, 12, 1000), dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)
    patient_id = np.zeros(n, dtype=np.int64)
    strat_fold = np.zeros(n, dtype=np.int64)
    ecg_id = np.zeros(n, dtype=np.int64)

    for i, path in enumerate(files):
        d = load_pilot_record(path)
        X[i] = d["X"]
        y[i] = int(d["y"])
        patient_id[i] = int(d["patient_id"])
        strat_fold[i] = int(d["strat_fold"])
        ecg_id[i] = int(d["ecg_id"])

    data = {"X": X, "y": y, "patient_id": patient_id,
            "strat_fold": strat_fold, "ecg_id": ecg_id}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **data)
    return data


def rr_features(positions, record_ids, fs=360):
    # Timing features for every beat, from the R-peak positions of its neighbours
    # in the SAME recording (so nothing is taken from other patients).
    # A single beat window cannot show if a beat came early, and early beats
    # are what separates S from N. Columns:
    #   0 pre-RR (s), 1 post-RR (s), 2 mean of the last 10 RR (s),
    #   3 pre-RR / local mean, 4 post-RR / local mean
    positions = np.asarray(positions, dtype=np.float64)
    record_ids = np.asarray(record_ids)
    feats = np.zeros((len(positions), 5), dtype=np.float32)

    for rec in np.unique(record_ids):
        rows = np.where(record_ids == rec)[0]
        pos = positions[rows]
        if np.any(np.diff(pos) <= 0):
            raise ValueError(f"beats of record {rec} are not in time order")
        rr = np.diff(pos) / fs
        pre = np.concatenate([[rr[0]], rr])    # first beat: reuse the next interval
        post = np.concatenate([rr, [rr[-1]]])  # last beat: reuse the previous one
        local = np.array([pre[max(0, i - 9):i + 1].mean() for i in range(len(pre))])
        feats[rows] = np.stack([pre, post, local, pre / local, post / local], axis=1)
    return feats


def ptbxl_split(strat_fold):
    # the PTB-XL authors recommend folds 1-8 train, 9 val, 10 test
    fold = np.asarray(strat_fold)
    return {"train": np.where(fold <= 8)[0],
            "val": np.where(fold == 9)[0],
            "test": np.where(fold == 10)[0]}


def mitbih_group_split(y, groups, val_frac=0.15, test_frac=0.15,
                       min_per_class=20, seed=42, max_tries=20000):
    # Split by patient group, not by beat. S and F beats sit in just a few
    # patients, so a random shuffle can leave val or test without a class.
    # Keep reshuffling until every part has at least min_per_class of each class.
    # Fixed seed -> same split every time.
    y = np.asarray(y)
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)
    n_val = max(1, round(len(unique_groups) * val_frac))
    n_test = max(1, round(len(unique_groups) * test_frac))

    rng = np.random.default_rng(seed)
    for _ in range(max_tries):
        shuffled = rng.permutation(unique_groups)
        chosen = {"test": shuffled[:n_test],
                  "val": shuffled[n_test:n_test + n_val],
                  "train": shuffled[n_test + n_val:]}
        split = {part: np.where(np.isin(groups, g))[0] for part, g in chosen.items()}
        ok = True
        for idx in split.values():
            if np.bincount(y[idx], minlength=len(MITBIH_CLASSES)).min() < min_per_class:
                ok = False
                break
        if ok:
            return split

    raise RuntimeError("no split found, try a smaller min_per_class")


def mitbih_cv_folds(y, groups, n_splits=5, seed=42):
    # k-fold where a patient group never ends up on both sides
    from sklearn.model_selection import StratifiedGroupKFold
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y, groups))


def assert_no_leakage(split, owner, name="patient"):
    # owner = patient id (or group) of every sample
    owner = np.asarray(owner)
    for i, a in enumerate(PARTS):
        for b in PARTS[i + 1:]:
            if set(split[a].tolist()) & set(split[b].tolist()):
                raise AssertionError(f"same sample in {a} and {b}")
            shared = set(owner[split[a]].tolist()) & set(owner[split[b]].tolist())
            if shared:
                raise AssertionError(f"{name} in both {a} and {b}: {sorted(shared)[:10]}")


def class_report(y, split, class_names):
    lines = [f"{'part':<6}{'n':>8}  " + "  ".join(f"{c:>7}" for c in class_names)]
    for part in PARTS:
        counts = Counter(y[split[part]].tolist())
        row = "  ".join(f"{counts.get(k, 0):>7}" for k in range(len(class_names)))
        lines.append(f"{part:<6}{len(split[part]):>8}  {row}")
    return "\n".join(lines)


def make_torch_dataset(X, y, idx):
    import torch
    from torch.utils.data import TensorDataset
    return TensorDataset(torch.from_numpy(X[idx]), torch.from_numpy(y[idx]))


def main():
    print("== MIT-BIH ==")
    m = load_mitbih()
    print("beats:", len(m["y"]), "groups:", len(np.unique(m["groups"])))
    split = mitbih_group_split(m["y"], m["groups"])
    assert_no_leakage(split, m["groups"], "patient group")
    print(class_report(m["y"], split, MITBIH_CLASSES))
    print("OK: no patient-group leakage")

    print("\n== PTB-XL ==")
    p = load_ptbxl()
    print("ECGs:", len(p["y"]), "patients:", len(np.unique(p["patient_id"])))
    split = ptbxl_split(p["strat_fold"])
    assert_no_leakage(split, p["patient_id"], "patient")
    print(class_report(p["y"], split, PTBXL_CLASSES))
    print("OK: no patient leakage")


if __name__ == "__main__":
    main()