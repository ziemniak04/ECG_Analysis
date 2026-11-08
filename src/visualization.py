"""Raw ECG plots with expert annotation symbols; no preprocessing."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .data_loader import ECGRecording

# WFDB beat symbols, including rare beat types. Other markers remain visible separately.
BEAT_SYMBOLS = frozenset({"N", "L", "R", "B", "A", "a", "J", "S", "V", "r", "F", "e", "j", "n", "E", "/", "f", "Q"})


def plot_recording(record: ECGRecording, start_seconds: float = 0, duration_seconds: float = 10):
    """Plot every channel, with beats red and non-beat annotations gray."""
    if not np.isfinite(start_seconds) or start_seconds < 0:
        raise ValueError("start_seconds must be finite and nonnegative")
    if not np.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("duration_seconds must be finite and positive")
    fs = record.sampling_frequency
    start = int(round(start_seconds * fs))
    stop = min(len(record.signals), int(round((start_seconds + duration_seconds) * fs)))
    if start >= stop:
        raise ValueError("Requested interval contains no samples")
    time = np.arange(start, stop) / fs
    fig, axes = plt.subplots(record.signals.shape[1], 1, sharex=True, squeeze=False,
                             figsize=(14, 3 * record.signals.shape[1]))
    selected = np.flatnonzero((record.annotation_positions >= start) & (record.annotation_positions < stop))
    for channel, ax in enumerate(axes[:, 0]):
        ax.plot(time, record.signals[start:stop, channel], linewidth=0.8)
        for index in selected:
            pos = record.annotation_positions[index]
            symbol = record.annotation_symbols[index]
            color = "tab:red" if symbol in BEAT_SYMBOLS else "gray"
            ax.scatter(pos / fs, record.signals[pos, channel], color=color, s=15, zorder=3)
            ax.annotate(symbol, (pos / fs, record.signals[pos, channel]),
                        xytext=(0, 9), textcoords="offset points", color=color, fontsize=8)
        ax.set_ylabel(f"{record.channel_names[channel]} ({record.units[channel]})")
        ax.grid(alpha=0.25)
    axes[-1, 0].set_xlabel("Time (s)")
    fig.suptitle(f"Record {record.record_id}: raw ECG | red: beats, gray: other annotations")
    fig.tight_layout()
    return fig


def plot_comparison(records: list[ECGRecording], duration_seconds: float = 10):
    """Compare the first channel of each recording on a shared time scale."""
    if not records or not np.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("Provide recordings and a positive finite duration")
    fig, axes = plt.subplots(len(records), 1, sharex=True, squeeze=False, figsize=(14, 3 * len(records)))
    for record, ax in zip(records, axes[:, 0]):
        n = min(len(record.signals), int(round(duration_seconds * record.sampling_frequency)))
        if n < 1:
            raise ValueError("Requested interval contains no samples")
        ax.plot(np.arange(n) / record.sampling_frequency, record.signals[:n, 0], linewidth=0.8)
        ax.set_ylabel(f"{record.record_id}: {record.channel_names[0]} ({record.units[0]})")
        ax.grid(alpha=0.25)
    axes[-1, 0].set_xlabel("Time (s)")
    fig.suptitle("Raw ECG comparison (first available channel)")
    fig.tight_layout()
    return fig


def save_figure(fig, path: Path | str) -> Path:
    """Save a new image, refusing to overwrite an existing file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as output:
        fig.savefig(output, format="png", dpi=150)
    return path
