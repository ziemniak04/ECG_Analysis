"""Synchronized offline ECG stage comparisons; no beat segmentation."""
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
import csv
import numpy as np
from .data_loader import ECGRecording
from .preprocessing import filter_recording, normalize_ecg
from .visualization import BEAT_SYMBOLS
import matplotlib.pyplot as plt

STAGES = ("raw", "filtered", "filtered_zscore")
TITLES = ("Raw ECG", "Filtered ECG", "Filtered + Z-score ECG")


@dataclass(frozen=True)
class PreprocessingComparison:
    """An illustrative excerpt, retaining original recording sample coordinates."""
    record_id: str
    sampling_frequency: float
    channel_names: tuple[str, ...]
    physical_units: tuple[str, ...]
    start_sample: int
    stop_sample: int
    time: np.ndarray
    stages: dict[str, np.ndarray]
    annotation_positions: np.ndarray
    annotation_symbols: tuple[str, ...]


def compare_recording(record: ECGRecording, start_seconds=10.0, duration_seconds=10.0):
    """Process the whole recording with existing defaults, then select an excerpt.

    Z-score statistics belong to this recording alone, independently per channel.
    This is a display interval, not a heartbeat window or training split.
    Only expert heartbeat symbols are selected; other annotation types are omitted.
    """
    for name, value in (("start_seconds", start_seconds), ("duration_seconds", duration_seconds)):
        if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
            raise ValueError(f"{name} must be a finite number")
    if start_seconds < 0 or duration_seconds <= 0:
        raise ValueError("Need nonnegative start and positive duration")
    fs = record.sampling_frequency
    start = int(round(start_seconds * fs))
    stop = int(round((start_seconds + duration_seconds) * fs))
    if start >= stop or stop > len(record.signals):
        raise ValueError("Requested complete interval is unavailable")
    filtered = filter_recording(record)
    normalized = normalize_ecg(filtered.signals, source_ids=[record.record_id],
                               channel_names=record.channel_names)
    selected = [i for i, (pos, symbol) in enumerate(zip(record.annotation_positions, record.annotation_symbols))
                if start <= pos < stop and symbol in BEAT_SYMBOLS]
    return PreprocessingComparison(
        record.record_id, fs, record.channel_names, record.units, start, stop,
        np.arange(start, stop) / fs,
        {"raw": record.signals[start:stop].copy(), "filtered": filtered.signals[start:stop].copy(),
         "filtered_zscore": normalized[start:stop].copy()},
        record.annotation_positions[selected].copy(), tuple(record.annotation_symbols[i] for i in selected),
    )


def comparison_statistics(comparison):
    """Describe exactly the displayed samples, using population std (ddof=0)."""
    rows = []
    for stage in STAGES:
        for channel, name in enumerate(comparison.channel_names):
            values = comparison.stages[stage][:, channel]
            rows.append({"record_id": comparison.record_id, "channel_index": channel,
                         "channel_name": name, "stage": stage,
                         "unit": "dimensionless" if stage == "filtered_zscore" else comparison.physical_units[channel],
                         "sampling_frequency_hz": comparison.sampling_frequency,
                         "start_sample": comparison.start_sample, "stop_sample_exclusive": comparison.stop_sample,
                         "start_seconds": comparison.start_sample / comparison.sampling_frequency,
                         "stop_seconds_exclusive": comparison.stop_sample / comparison.sampling_frequency,
                         "sample_count": len(values), "statistics_scope": "displayed_fragment",
                         "normalization_scope": "recording_per_channel" if stage == "filtered_zscore" else "none",
                         "mean": float(values.mean()), "std": float(values.std(ddof=0)),
                         "min": float(values.min()), "max": float(values.max())})
    return rows


def plot_preprocessing_comparison(comparison):
    """Plot stages on identical absolute time axes, without relocating annotations."""
    channels = len(comparison.channel_names)
    fig, axes = plt.subplots(channels, 3, sharex=True, squeeze=False, figsize=(18, 4 * channels))
    beat_times = comparison.annotation_positions / comparison.sampling_frequency
    indices = comparison.annotation_positions - comparison.start_sample
    for channel, name in enumerate(comparison.channel_names):
        for column, stage in enumerate(STAGES):
            ax = axes[channel, column]
            values = comparison.stages[stage][:, channel]
            ax.plot(comparison.time, values, linewidth=0.75, color=("tab:blue", "tab:orange", "tab:green")[column])
            ax.scatter(beat_times, values[indices], s=12, color="tab:red", zorder=3)
            for x, y, symbol in zip(beat_times, values[indices], comparison.annotation_symbols):
                ax.annotate(symbol, (x, y), xytext=(0, 7), textcoords="offset points", fontsize=7,
                            color="tab:red", annotation_clip=True)
            unit = "dimensionless" if stage == "filtered_zscore" else comparison.physical_units[channel]
            ax.set_title(TITLES[column])
            ax.set_ylabel(f"{name} ({unit})")
            ax.set_xlabel("Time from recording start (s)")
            ax.set_xlim(comparison.start_sample / comparison.sampling_frequency,
                        comparison.stop_sample / comparison.sampling_frequency)
            ax.grid(alpha=0.25)
            ax.margins(y=0.18)
    fig.suptitle(f"Record {comparison.record_id} | fs = {comparison.sampling_frequency:g} Hz | "
                 "red: expert heartbeat positions (unchanged)")
    fig.tight_layout()
    return fig


def save_comparison_figure(fig, path):
    """Save a 300 dpi PNG, refusing to overwrite an existing file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as output:
        fig.savefig(output, format="png", dpi=300)
    return path


def save_comparison_statistics(rows, path):
    """Write a new CSV, retaining record IDs as textual identifiers."""
    if not rows:
        raise ValueError("No comparison statistics to save")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path
