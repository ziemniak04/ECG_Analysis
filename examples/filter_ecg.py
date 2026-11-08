"""Filter full local records, then plot interior excerpts (no normalization)."""
import argparse
import json
from pathlib import Path
from src.data_loader import DEFAULT_DATA_DIR, load_recording
from src.preprocessing import filter_recording, powerline_diagnostics
from src.visualization import save_figure
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("results/preprocessing/phase4"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"parameters": {"highpass_hz": 0.5, "lowpass_hz": 45, "order": 4,
                              "notch_hz": None, "zero_phase": True}, "records": {}}
    for rid in ("100", "101", "200"):
        raw = load_recording(rid, args.data_dir)
        filtered = filter_recording(raw)
        fs = raw.sampling_frequency
        diagnostics = {}
        for hz in (50, 60):
            diagnostics[str(hz)] = {"raw": powerline_diagnostics(raw.signals, fs, hz),
                                    "filtered_without_notch": powerline_diagnostics(filtered.signals, fs, hz)}
        report["records"][rid] = diagnostics
        # Filter the full recording first; avoid boundary transients in excerpts.
        for start, duration, label in ((10, 10, "10s"), (10, 3, "detail")):
            lo, hi = int(start * fs), int((start + duration) * fs)
            time = np.arange(lo, hi) / fs
            fig, axes = plt.subplots(raw.signals.shape[1], 1, sharex=True, squeeze=False, figsize=(14, 7))
            for channel, ax in enumerate(axes[:, 0]):
                ax.plot(time, raw.signals[lo:hi, channel], label="Raw", alpha=0.7, linewidth=0.9)
                ax.plot(time, filtered.signals[lo:hi, channel], label="Filtered: 0.5–45 Hz, no notch", linewidth=0.9)
                ax.set_ylabel(f"{raw.channel_names[channel]} ({raw.units[channel]})")
                ax.grid(alpha=0.25)
                ax.legend(loc="upper right")
            axes[-1, 0].set_xlabel("Time (s)")
            fig.suptitle(f"Record {rid}: raw vs offline zero-phase filtered ECG")
            fig.tight_layout()
            try:
                print(save_figure(fig, args.output_dir / f"record_{rid}_raw_filtered_{label}.png"))
            finally:
                plt.close(fig)
    with (args.output_dir / "powerline_diagnostics.json").open("x") as output:
        json.dump(report, output, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
