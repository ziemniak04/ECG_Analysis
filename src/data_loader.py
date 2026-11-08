"""Local-only MIT-BIH loading. Signals use [samples, channels]."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import wfdb

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data/mitdb/physionet.org/files/mitdb/1.0.0"

@dataclass(frozen=True)
class ECGRecording:
    """Physical ECG samples and unmodified expert annotations."""
    record_id: str
    signals: np.ndarray
    sampling_frequency: float
    channel_names: tuple[str, ...]
    units: tuple[str, ...]
    annotation_positions: np.ndarray
    annotation_symbols: tuple[str, ...]
    annotation_aux_notes: tuple[str, ...]


def locate_data_dir(root: Path | str = DEFAULT_DATA_DIR) -> Path:
    """Find one recording directory, rejecting ambiguous recursive matches."""
    root = Path(root).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    directories = {p.parent for p in root.rglob("[0-9][0-9][0-9].hea")}
    if not directories:
        raise FileNotFoundError(f"No MIT-BIH headers found under {root}")
    if len(directories) != 1:
        raise ValueError(f"Multiple recording directories under {root}: {sorted(directories)}")
    return directories.pop()


def inspect_recordings(root: Path | str = DEFAULT_DATA_DIR) -> dict[str, tuple[str, ...]]:
    """Report missing companions, including orphan annotation/data files."""
    directory = locate_data_dir(root)
    ids = {p.stem for ext in ("hea", "dat", "atr") for p in directory.glob(f"*.{ext}")
           if len(p.stem) == 3 and p.stem.isdigit()}
    return {rid: tuple(ext for ext in ("hea", "dat", "atr")
                       if not (directory / f"{rid}.{ext}").is_file()) for rid in sorted(ids)}


def list_recordings(root: Path | str = DEFAULT_DATA_DIR) -> list[str]:
    """List numeric recording IDs with all three required files present."""
    return [rid for rid, missing in inspect_recordings(root).items() if not missing]


def load_recording(record_id: str = "100", root: Path | str = DEFAULT_DATA_DIR) -> ECGRecording:
    """Read one complete recording and atr annotations without network access."""
    record_id = str(record_id)
    if len(record_id) != 3 or not record_id.isascii() or not record_id.isdigit():
        raise ValueError("record_id must be a three-digit MIT-BIH ID")
    directory = locate_data_dir(root)
    for ext in ("hea", "dat", "atr"):
        path = directory / f"{record_id}.{ext}"
        if not path.is_file():
            raise FileNotFoundError(f"Required local file missing: {path}")
    base = str(directory / record_id)
    try:
        record = wfdb.rdrecord(base)
        annotation = wfdb.rdann(base, "atr")
    except Exception as exc:
        raise ValueError(f"Cannot read local recording {record_id}: {exc}") from exc
    signals = np.asarray(record.p_signal, dtype=np.float64)
    if signals.ndim != 2 or signals.shape != (record.sig_len, record.n_sig) or not signals.size:
        raise ValueError(f"Invalid signal dimensions for {record_id}: {signals.shape}")
    if not np.isfinite(signals).all():
        raise ValueError(f"Non-finite signal samples in {record_id}")
    if not np.isfinite(record.fs) or record.fs <= 0:
        raise ValueError("Sampling frequency must be positive and finite")
    if len(record.sig_name) != signals.shape[1] or len(record.units) != signals.shape[1]:
        raise ValueError("Channel metadata does not match signals")
    positions = np.asarray(annotation.sample, dtype=np.int64)
    if positions.ndim != 1 or len(positions) != len(annotation.symbol):
        raise ValueError("Annotation positions and symbols do not match")
    if np.any(positions < 0) or np.any(positions >= len(signals)) or np.any(np.diff(positions) < 0):
        raise ValueError("Invalid annotation positions")
    return ECGRecording(record_id, signals, float(record.fs), tuple(record.sig_name),
                        tuple(record.units), positions, tuple(annotation.symbol), tuple(annotation.aux_note))
