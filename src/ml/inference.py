"""Everything the Streamlit app needs to run the models on a new recording.

No streamlit in here, so it can be tested on its own. The preprocessing has to
be the same as in training (preprocessing code is reused for that):
  1. filter the whole recording (high-pass 0.5 Hz, low-pass 45 Hz, order 4, zero-phase, no notch)
  2. z-score every lead
  3. MIT-BIH: cut a window of 90 samples before / 162 after every R-peak
     PTB-XL:   cut 10 s windows (1000 samples at 100 Hz)
"""
import io
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

from src.ml.data import CACHE_DIR, MITBIH_CLASSES, PTBXL_CLASSES, ROOT, rr_features
from src.ml.models import ECGNet
from src.preprocessing import filter_ecg, normalize_ecg

MODELS_DIR = ROOT / "models"
MITBIH_FS = 360
PTBXL_FS = 100
BEFORE, AFTER = 90, 162          # samples around the R-peak (0.25 s / 0.45 s at 360 Hz)
RHYTHM_WINDOW = 1000             # 10 s at 100 Hz
PTBXL_LEADS = ("I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6")

# models
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

def load_model(name):
    ckpt = torch.load(MODELS_DIR / f"{name}.pt", map_location="cpu")
    n_extra = ckpt.get("n_extra", 0)  # old checkpoints have no RR features
    model = ECGNet(ckpt["in_channels"], ckpt["n_classes"], n_extra=n_extra)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return LoadedModel(name, model, list(ckpt["class_names"]), n_extra,
                       np.array(ckpt.get("extra_mean", []), dtype=np.float32),
                       np.array(ckpt.get("extra_std", []), dtype=np.float32),
                       int(ckpt["epoch"]), int(ckpt["in_channels"]))

def list_models():
    # {"mitbih": 2, "mitbih_rr": 2, "ptbxl": 12} -> name: number of input leads
    found = {}
    for path in sorted(MODELS_DIR.glob("*.pt")):
        try:
            found[path.stem] = int(torch.load(path, map_location="cpu")["in_channels"])
        except Exception:
            pass  # not one of our checkpoints
    return found

def _predict(loaded, X, extra, batch_size=512):
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i:i + batch_size])
            eb = torch.from_numpy(extra[i:i + batch_size])
            out.append(torch.softmax(loaded.model(xb, eb), dim=1).numpy())
    return np.concatenate(out)

# reading files
def _rec_to_dict(rec, source):
    if rec.p_signal is None:
        raise ValueError("The record has no physical signal")
    return {"signal": np.asarray(rec.p_signal, dtype=np.float64), "fs": float(rec.fs),
            "names": list(rec.sig_name), "source": source}

def read_wfdb_path(path):
    """A record on disk, path with or without the .hea ending."""
    path = Path(path).with_suffix("")
    return _rec_to_dict(wfdb.rdrecord(str(path)), path.name)

def read_wfdb_files(files):
    """files: {file name: bytes}. Needs the .hea and the .dat of ONE record."""
    with tempfile.TemporaryDirectory() as tmp:
        for name, data in files.items():
            (Path(tmp) / Path(name).name).write_bytes(data)  # Path(...).name drops any folders
        headers = sorted(Path(tmp).glob("*.hea"))
        if len(headers) != 1:
            raise ValueError("Upload exactly one .hea file together with its .dat file")
        try:
            rec = wfdb.rdrecord(str(headers[0].with_suffix("")))
        except Exception as e:
            raise ValueError(f"Could not read the record, is the .dat file missing? ({e})")
        return _rec_to_dict(rec, headers[0].stem)

def read_csv_file(data, name="uploaded.csv"):
    """Columns = leads, rows = samples. A time/index column is dropped if it has a typical name."""
    try:
        df = pd.read_csv(io.BytesIO(data), sep=None, engine="python")
    except Exception as e:
        raise ValueError(f"Could not read the csv file ({e})")
    # no header: the first row is numbers, pandas then took it as column names
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
    """Put the 12 leads in the order used in training. Returns (signal, note)."""
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

# preprocessing
def prepare(signal, fs):
    """Same as in training: filter the whole recording, then z-score per lead."""
    filtered = filter_ecg(signal, fs)
    normalized = normalize_ecg(filtered, "zscore", scope="recording", source_ids=["upload"])
    return filtered, normalized

# beat model
def detect_peaks(normalized, fs=MITBIH_FS):
    # xqrs finds the QRS complexes, then every peak is moved to the local extreme
    lead = normalized[:, 0]
    peaks = processing.xqrs_detect(lead, fs, verbose=False)
    if len(peaks) < 3:
        raise ValueError("Fewer than 3 heartbeats found, the recording is too short or too noisy")
    peaks = processing.correct_peaks(lead, np.asarray(peaks), search_radius=int(0.05 * fs),
                                     smooth_window_size=int(0.02 * fs))
    return np.unique(np.asarray(peaks, dtype=np.int64))

def classify_beats(loaded, normalized, peaks):
    # RR features use ALL detected beats, only then the beats too close to the edge are dropped
    if loaded.n_extra:
        extra = rr_features(peaks, np.zeros(len(peaks)), MITBIH_FS)
        extra = ((extra - loaded.extra_mean) / loaded.extra_std).astype(np.float32)
    else:
        extra = np.zeros((len(peaks), 0), dtype=np.float32)

    keep = (peaks - BEFORE >= 0) & (peaks + AFTER <= len(normalized))
    peaks, extra = peaks[keep], extra[keep]
    X = np.stack([normalized[p - BEFORE:p + AFTER].T for p in peaks]).astype(np.float32)

    proba = _predict(loaded, X, extra)
    return {"peaks": peaks, "proba": proba, "pred": proba.argmax(1)}

# rhythm model
def classify_rhythm(loaded, filtered):
    """filtered: (samples, 12) at 100 Hz. Non-overlapping 10 s windows, z-score per window."""
    n_windows = len(filtered) // RHYTHM_WINDOW
    if n_windows < 1:
        raise ValueError(f"Need at least 10 s of signal at 100 Hz, got {len(filtered) / PTBXL_FS:.1f} s")
    X = []
    for w in range(n_windows):
        part = filtered[w * RHYTHM_WINDOW:(w + 1) * RHYTHM_WINDOW]
        z = normalize_ecg(part, "zscore", scope="recording", source_ids=["window"])
        X.append(np.ascontiguousarray(z.T, dtype=np.float32))
    X = np.stack(X)
    proba = _predict(loaded, X, np.zeros((len(X), 0), dtype=np.float32))
    return {"starts": np.arange(n_windows) * RHYTHM_WINDOW, "proba": proba,
            "pred": proba.argmax(1), "mean_proba": proba.mean(0),
            "unused_samples": len(filtered) - n_windows * RHYTHM_WINDOW}


# which split was a local record in?
def split_of_mitbih_record(record_id):
    """'train' / 'val' / 'test' for a MIT-BIH record, or None if unknown (no cache yet)."""
    try:
        from src.ml.data import mitbih_group_split
        with np.load(CACHE_DIR / "mitbih_v2.npz", allow_pickle=False) as f:
            y, groups, rec = f["y"], f["groups"], f["record_ids"]
        split = mitbih_group_split(y, groups)
        for part, idx in split.items():
            if str(record_id) in set(rec[idx].tolist()):
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