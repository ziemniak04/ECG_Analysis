"""Reproduce three-stage ECG comparisons from local recordings only."""
import argparse
from pathlib import Path
from src.data_loader import DEFAULT_DATA_DIR, load_recording
from src.preprocessing_comparison import (compare_recording, comparison_statistics,
    plot_preprocessing_comparison, save_comparison_figure, save_comparison_statistics)
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("results/preprocessing/phase6"))
    parser.add_argument("--start-seconds", type=float, default=10.0)
    parser.add_argument("--duration-seconds", type=float, default=10.0)
    args = parser.parse_args()
    rows = []
    for rid in ("100", "101", "200"):
        record = load_recording(rid, args.data_dir)
        comparison = compare_recording(record, args.start_seconds, args.duration_seconds)
        fig = plot_preprocessing_comparison(comparison)
        try:
            print(save_comparison_figure(fig, args.output_dir / f"record_{rid}_preprocessing_comparison.png"))
        finally:
            plt.close(fig)
        rows.extend(comparison_statistics(comparison))
    print(save_comparison_statistics(rows, args.output_dir / "preprocessing_statistics.csv"))


if __name__ == "__main__":
    main()
