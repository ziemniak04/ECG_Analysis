"""Inference pipeline for the Streamlit ECG viewer.

Reuses the exact filtering and z-score standardization functions from training
to prevent training-serving feature skew.

References
----------
.. [1] Sculley, D., Holt, G., Golovin, D., Davydov, E., Phillips, T., Ebner, D.,
       Chaudhary, V., Young, M., Crespo, J. F., & Dennison, D. (2015).
       Hidden technical debt in machine learning systems. Advances in Neural
       Information Processing Systems (NeurIPS), 28.
       Access: https://papers.nips.cc/paper_files/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html
.. [2] Strodthoff, N., Wagner, P., Schaeffter, T., & Samek, W. (2020).
       Deep learning for ECG analysis: Benchmarks and insights from PTB-XL.
       IEEE Journal of Biomedical and Health Informatics, 25(5), 1519-1528.
       Access: https://arxiv.org/abs/2004.13701

       I took the first articles advice of "reusing the exact preprocessing functions from training" to avoid
       training-serving skew. And the second one I used to get the idea of how to preprocess the PTB-XL data,
       and also to get the lead names and the rhythm classes.
"""

import io
import json
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wfdb
from scipy.signal import resample_poly
from wfdb import processing

from functools import lru_cache

from src.ml.data import CACHE_DIR, MITBIH_CLASSES, PTBXL_CLASSES, ROOT, rr_features
from src.ml.evaluate import apply_thresholds, validate_checkpoint
from src.ml.models import ECGNet
from src.preprocessing import filter_ecg, normalize_ecg

MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results" / "ml"
MITBIH_FS = 360
PTBXL_FS = 100
BEFORE, AFTER = 90, 162
RHYTHM_WINDOW = 1000
PTBXL_LEADS = ("I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6")


@dataclass
class LoadedModel:
    name: str
    model: torch.nn.Module
    class_names: list
    n_extra: int
    extra_mean: np.ndarray
    extra_std: np.ndarray
    epoch: int
    in_channels: int
    thresholds: np.ndarray | None = None


def load_model(name):
    ckpt = torch.load(MODELS_DIR / f"{name}.pt", map_location="cpu", weights_only=False)
    validate_checkpoint(ckpt)
    n_extra = ckpt.get("n_extra", 0)
    class_names = list(ckpt["class_names"])
    model = ECGNet(ckpt["in_channels"], ckpt["n_classes"], n_extra=n_extra)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    thresholds = None
    if ckpt.get("thresholds") is not None:
        thresholds = np.array(ckpt["thresholds"], dtype=np.float32)
    else:
        th_path = RESULTS_DIR / f"{name}_thresholds.json"
        if th_path.is_file():
            saved = json.loads(th_path.read_text())
            thresholds = np.array([saved[c] for c in class_names], dtype=np.float32)

    return LoadedModel(
        name,
        model,
        class_names,
        n_extra,
        np.array(ckpt.get("extra_mean", []), dtype=np.float32),
        np.array(ckpt.get("extra_std", []), dtype=np.float32),
        int(ckpt["epoch"]),
        int(ckpt["in_channels"]),
        thresholds,
    )


def list_models():
    found = {}
    for path in sorted(MODELS_DIR.glob("*.pt")):
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            validate_checkpoint(ckpt)
            found[path.stem] = int(ckpt["in_channels"])
        except Exception:
            pass
    return found


def _predict(loaded, X, extra, batch_size=512):
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i:i + batch_size])
            eb = torch.from_numpy(extra[i:i + batch_size])
            out.append(torch.softmax(loaded.model(xb, eb), dim=1).numpy())
    return np.concatenate(out)


def _rec_to_dict(rec, source):
    if rec.p_signal is None:
        raise ValueError("The record has no physical signal")
    sig = np.asarray(rec.p_signal, dtype=np.float64)
    if not np.isfinite(sig).all():
        raise ValueError("The recording contains NaN or infinite signal values")
    return {"signal": sig, "fs": float(rec.fs), "names": list(rec.sig_name), "source": source}


def read_wfdb_path(path):
    path = Path(path).with_suffix("")
    return _rec_to_dict(wfdb.rdrecord(str(path)), path.name)


def read_wfdb_files(files):
    with tempfile.TemporaryDirectory() as tmp:
        for name, data in files.items():
            (Path(tmp) / Path(name).name).write_bytes(data)
        headers = sorted(Path(tmp).glob("*.hea"))
        if len(headers) != 1:
            raise ValueError("Upload exactly one .hea file together with its .dat file")
        try:
            rec = wfdb.rdrecord(str(headers[0].with_suffix("")))
        except Exception as e:
            raise ValueError(f"Could not read the record, is the .dat file missing? ({e})")
        return _rec_to_dict(rec, headers[0].stem)


def read_csv_file(data, name="uploaded.csv"):
    try:
        df = pd.read_csv(io.BytesIO(data), sep=None, engine="python")
    except Exception as e:
        raise ValueError(f"Could not read the csv file ({e})")
    try:
        [float(c) for c in df.columns]
        df = pd.read_csv(io.BytesIO(data), sep=None, engine="python", header=None)
        names = None
    except ValueError:
        names = [str(c).strip() for c in df.columns]
    if names and names[0].lower() in ("time", "t", "sample", "samples", "index", "time_s"):
        df = df.iloc[:, 1:]
        names = names[1:]
    df = df.apply(pd.to_numeric, errors="coerce")
    if df.isna().any().any():
        raise ValueError(f"The csv contains {int(df.isna().sum().sum())} empty or non-numeric values")
    return {"signal": df.to_numpy(dtype=np.float64), "fs": None, "names": names, "source": Path(name).stem}


def order_leads(signal, names):
    if names is None:
        if signal.shape[1] != 12:
            raise ValueError(f"The rhythm model needs 12 leads, the file has {signal.shape[1]}")
        return signal, "No lead names in the file, assuming the order I, II, III, aVR, aVL, aVF, V1-V6."
    upper = [n.strip().upper() for n in names]
    if sorted(upper) != sorted(PTBXL_LEADS):
        raise ValueError(f"Lead names {names} do not match the 12 standard leads")
    if upper == list(PTBXL_LEADS):
        return signal, ""
    order = [upper.index(l) for l in PTBXL_LEADS]
    return signal[:, order], "The leads were in a different order and have been reordered."


def resample(signal, fs_in, fs_out):
    if abs(fs_in - fs_out) < 1e-9:
        return signal
    ratio = Fraction(fs_out / fs_in).limit_denominator(1000)
    return resample_poly(signal, ratio.numerator, ratio.denominator, axis=0)


def prepare(signal, fs):
    if not np.isfinite(signal).all():
        raise ValueError("Signal contains NaN or infinite values.")
    if np.any(np.std(signal, axis=0) < 1e-6):
        raise ValueError("One or more leads appear to be flatlined (zero variance).")
    filtered = filter_ecg(signal, fs)
    normalized = normalize_ecg(filtered, "zscore", scope="recording", source_ids=["upload"])
    return filtered, normalized


def detect_peaks(normalized, fs=MITBIH_FS):
    lead = normalized[:, 0]
    peaks = processing.xqrs_detect(lead, fs, verbose=False)
    if len(peaks) < 3:
        raise ValueError("Fewer than 3 heartbeats found, the recording is too short or too noisy")
    peaks = processing.correct_peaks(
        lead, np.asarray(peaks), search_radius=int(0.05 * fs), smooth_window_size=int(0.02 * fs)
    )
    return np.unique(np.asarray(peaks, dtype=np.int64))


def classify_beats(loaded, normalized, peaks):
    if loaded.n_extra:
        extra = rr_features(peaks, np.zeros(len(peaks)), MITBIH_FS)
        extra = ((extra - loaded.extra_mean) / loaded.extra_std).astype(np.float32)
    else:
        extra = np.zeros((len(peaks), 0), dtype=np.float32)

    keep = (peaks - BEFORE >= 0) & (peaks + AFTER <= len(normalized))
    peaks, extra = peaks[keep], extra[keep]
    X = np.stack([normalized[p - BEFORE:p + AFTER].T for p in peaks]).astype(np.float32)

    proba = _predict(loaded, X, extra)
    pred = apply_thresholds(proba, loaded.thresholds)
    return {"peaks": peaks, "proba": proba, "pred": pred}


def classify_rhythm(loaded, filtered):
    n_windows = len(filtered) // RHYTHM_WINDOW
    if n_windows < 1:
        raise ValueError(f"Need at least 10 s of signal at 100 Hz, got {len(filtered) / PTBXL_FS:.1f} s")
    X = []
    for w in range(n_windows):
        part = filtered[w * RHYTHM_WINDOW:(w + 1) * RHYTHM_WINDOW]
        z = normalize_ecg(part, "zscore", scope="recording", source_ids=["window"])
        if not np.isfinite(z).all():
            raise ValueError(f"10 s window starting at {w * 10} s contains a flatline or NaN.")        
        X.append(np.ascontiguousarray(z.T, dtype=np.float32))
    X = np.stack(X)
    proba = _predict(loaded, X, np.zeros((len(X), 0), dtype=np.float32))
    pred = apply_thresholds(proba, loaded.thresholds)
    return {
        "starts": np.arange(n_windows) * RHYTHM_WINDOW,
        "proba": proba,
        "pred": pred,
        "mean_proba": proba.mean(0),
        "unused_samples": len(filtered) - n_windows * RHYTHM_WINDOW,
    }


@lru_cache(maxsize=1)
def _mitbih_record_splits():
    from src.ml.data import mitbih_group_split
    with np.load(CACHE_DIR / "mitbih_v2.npz", allow_pickle=False) as f:
        y, groups, rec = f["y"], f["groups"], f["record_ids"]
    split = mitbih_group_split(y, groups)
    return {part: set(rec[idx].tolist()) for part, idx in split.items()}


def split_of_mitbih_record(record_id):
    try:
        for part, records in _mitbih_record_splits().items():
            if str(record_id) in records:
                return part
    except Exception:
        pass
    return None


def split_of_ptbxl_ecg(ecg_id):
    try:
        with np.load(CACHE_DIR / "ptbxl_v1.npz", allow_pickle=False) as f:
            ids, fold = f["ecg_id"], f["strat_fold"]
        hit = np.where(ids == int(ecg_id))[0]
        if len(hit) == 0:
            return None
        return "train" if fold[hit[0]] <= 8 else "val" if fold[hit[0]] == 9 else "test"
    except Exception:
        return None