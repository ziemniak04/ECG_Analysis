"""Run from the project root: python -m examples.inspect_ecg."""
import argparse
from collections import Counter
from pathlib import Path
from src.data_loader import DEFAULT_DATA_DIR, inspect_recordings, list_recordings, load_recording
from src.visualization import plot_recording, plot_comparison, save_figure
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("results/preprocessing"))
    args = parser.parse_args()
    available = list_recordings(args.data_dir)
    print("Complete recordings:", len(available), available)
    print("Incomplete recordings:", {k:v for k,v in inspect_recordings(args.data_dir).items() if v})
    record = load_recording("100", args.data_dir)
    print("Record:", record.record_id)
    print("Signal shape [samples, channels]:", record.signals.shape)
    print("Sampling frequency (Hz):", record.sampling_frequency)
    print("Duration (s):", len(record.signals) / record.sampling_frequency)
    print("Channels:", record.channel_names, "Units:", record.units)
    print("Annotations:", len(record.annotation_positions), dict(Counter(record.annotation_symbols)))
    fig = plot_recording(record)
    try:
        print("Saved:", save_figure(fig, args.output_dir / "record_100_first_10s.png"))
    finally:
        plt.close(fig)
    records = [record] + [load_recording(rid, args.data_dir) for rid in ("101", "200")]
    fig = plot_comparison(records)
    try:
        print("Saved:", save_figure(fig, args.output_dir / "records_100_101_200_comparison.png"))
    finally:
        plt.close(fig)

if __name__ == "__main__":
    main()
