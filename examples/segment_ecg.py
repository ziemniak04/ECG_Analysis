"""Reproduce annotation-based heartbeat segmentation, plots and diagnostic counts."""
import argparse
import json
from pathlib import Path
import numpy as np
from src.data_loader import DEFAULT_DATA_DIR, load_recording
from src.segmentation import segment_recording
from src.visualization import plot_recording
from src.preprocessing_comparison import save_comparison_figure
import matplotlib.pyplot as plt


def plot_example_segments(segments, beat_indices):
    """Plot both channels of selected windows relative to the expert annotation."""
    indices = list(beat_indices)
    if not indices or any(i < 0 or i >= len(segments.X) for i in indices):
        raise ValueError('Provide valid indices of extracted beats')
    channels = segments.X.shape[1]
    fig, axes = plt.subplots(channels, len(indices), sharex=True, squeeze=False,
                             figsize=(5*len(indices), 3.5*channels))
    time = (np.arange(segments.X.shape[2])-segments.before_samples)/segments.sampling_frequency
    for column, index in enumerate(indices):
        for channel in range(channels):
            ax = axes[channel,column]
            values = segments.X[index,channel]
            ax.plot(time, values, linewidth=1)
            ax.axvline(0,color='tab:red',linestyle='--',linewidth=0.8)
            ax.scatter([0],[values[segments.before_samples]],color='tab:red',zorder=3)
            ax.set_title(f'Original symbol: {segments.annotation_symbols[index]} | sample {segments.annotation_positions[index]}')
            ax.set_ylabel(f'{segments.channel_names[channel]} ({segments.units[channel]})')
            ax.set_xlabel('Time relative to expert annotation (s)')
            ax.grid(alpha=0.25)
    fig.suptitle(f'Record {segments.record_ids[indices[0]]}: filtered + per-record Z-score heartbeat windows | '
                 f'fs={segments.sampling_frequency:g} Hz | {segments.X.shape[2]} samples')
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=DEFAULT_DATA_DIR)
    parser.add_argument('--output-dir',type=Path,default=Path('results/preprocessing/phase7'))
    parser.add_argument('--before-seconds',type=float,default=0.25)
    parser.add_argument('--after-seconds',type=float,default=0.45)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    summary={}
    for rid in ('100','101','200'):
        raw=load_recording(rid,args.data_dir)
        segments=segment_recording(raw,before_seconds=args.before_seconds,after_seconds=args.after_seconds)
        indices=np.flatnonzero((segments.annotation_positions>=10*raw.sampling_frequency)&
                               (segments.annotation_positions<20*raw.sampling_frequency))[:3]
        fig=plot_recording(raw,start_seconds=10,duration_seconds=10)
        try:
            print(save_comparison_figure(fig,args.output_dir/f'record_{rid}_raw_annotations.png'))
        finally:
            plt.close(fig)
        fig=plot_example_segments(segments,indices)
        try:
            print(save_comparison_figure(fig,args.output_dir/f'record_{rid}_heartbeat_segments.png'))
        finally:
            plt.close(fig)
        summary[rid]={'shape':list(segments.X.shape),'report':segments.report,
                      'configuration':segments.configuration,'channel_names':list(segments.channel_names),
                      'patient_group_id':None,'plotted_beat_indices':indices.tolist(),
                      'plotted_annotation_positions':segments.annotation_positions[indices].tolist()}
        print(rid, segments.X.shape, segments.report)
    # Diagnostic metadata only; final dataset export is intentionally a later phase.
    with (args.output_dir/'segmentation_summary.json').open('x') as output:
        json.dump(summary,output,indent=2,allow_nan=False)


if __name__=='__main__':
    main()
