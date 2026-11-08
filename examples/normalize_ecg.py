"""Plot filtered and normalized ECG using separate per-record/channel statistics."""
import argparse
import json
from pathlib import Path
import numpy as np
from src.data_loader import DEFAULT_DATA_DIR, load_recording
from src.preprocessing import ECGNormalizer, filter_recording
from src.visualization import save_figure
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument('--output-dir', type=Path, default=Path('results/preprocessing/phase5'))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {'scope': 'recording', 'default_method': 'zscore',
              'filter': {'highpass_hz': 0.5, 'lowpass_hz': 45, 'order': 4, 'notch_hz': None},
              'records': {}}
    for rid in ('100', '101', '200'):
        record = filter_recording(load_recording(rid, args.data_dir))
        fs = record.sampling_frequency
        lo, hi = int(10*fs), int(13*fs)
        time = np.arange(lo, hi)/fs
        channels = record.signals.shape[1]
        fig, axes = plt.subplots(channels, 3, sharex=True, squeeze=False, figsize=(17, 3.5*channels))
        report['records'][rid] = {}
        try:
            for channel in range(channels):
                axes[channel, 0].plot(time, record.signals[lo:hi, channel], linewidth=0.9)
                axes[channel, 0].set_ylabel(f'{record.channel_names[channel]} (mV)')
                axes[channel, 0].set_title('Filtered, before normalization')
            for column, method in enumerate(('zscore', 'minmax'), 1):
                scaler = ECGNormalizer(method).fit(record.signals, scope='recording', source_ids=[rid],
                                                    channel_names=record.channel_names)
                normalized = scaler.transform(record.signals, channel_names=record.channel_names)
                scaler.save(args.output_dir/f'record_{rid}_{method}_parameters.json')
                # Verify exact inference reproducibility from saved parameters.
                restored = ECGNormalizer.load(args.output_dir/f'record_{rid}_{method}_parameters.json')
                np.testing.assert_array_equal(restored.transform(record.signals, channel_names=record.channel_names), normalized)
                report['records'][rid][method] = {'mean': normalized.mean(axis=0).tolist(),
                    'std': normalized.std(axis=0).tolist(), 'min': normalized.min(axis=0).tolist(),
                    'max': normalized.max(axis=0).tolist(), 'shape': list(normalized.shape),
                    'inference_roundtrip_exact': True}
                for channel in range(channels):
                    axes[channel, column].plot(time, normalized[lo:hi, channel], linewidth=0.9)
                    axes[channel, column].set_title('Z-score' if method == 'zscore' else 'Min-Max [0, 1]')
                    axes[channel, column].set_ylabel(f'{record.channel_names[channel]} (dimensionless)')
            for ax in axes.flat:
                ax.grid(alpha=0.25)
                ax.set_xlabel('Time (s)')
            fig.suptitle(f'Record {rid}: per-recording normalization after filtering (statistics from full recording)')
            fig.tight_layout()
            print(save_figure(fig, args.output_dir/f'record_{rid}_normalization.png'))
        finally:
            plt.close(fig)
    with (args.output_dir/'normalization_summary.json').open('x') as output:
        json.dump(report, output, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
