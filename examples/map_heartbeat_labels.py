"""Reproduce AAMI-style counts and real examples; no final dataset export."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from src.data_loader import DEFAULT_DATA_DIR, load_recording
from src.segmentation import segment_recording
from src.label_mapping import CLASS_NAMES, CLASS_INDICES, map_segments, summarize_recordings
from src.preprocessing_comparison import save_comparison_figure
import matplotlib.pyplot as plt


def save_csv(rows, path):
    with path.open('x', newline='', encoding='utf-8') as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument('--output-dir', type=Path, default=Path('results/preprocessing/phase8'))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ids = ['100','101','200']
    summary = summarize_recordings(args.data_dir, ids)
    class_rows, symbol_rows = [], []
    for rid, data in [*summary['records'].items(), ('ALL_SELECTED', summary['aggregate'])]:
        for name in CLASS_NAMES:
            class_rows.append({'record_id':rid, 'aami_class':name, 'class_index':CLASS_INDICES[name],
                'count':data['class_counts'][name], 'percent_of_mapped':data['class_percentages_of_mapped'][name],
                'percent_of_segments':data['class_percentages_of_segments'][name],
                'total_segments':data['segment_count'], 'unmapped_segments':data['unmapped_count']})
    for rid, data in summary['records'].items():
        raw_counts = data['original_all_annotation_symbol_counts']
        kept_counts = data['original_segment_symbol_counts']
        for symbol in sorted(raw_counts):
            symbol_rows.append({'record_id':rid, 'original_symbol':symbol, 'all_annotations':raw_counts[symbol],
                'segmented_beats':kept_counts.get(symbol,0),
                'aami_class':summary['mapping'].get(symbol,'UNMAPPED_OR_NONBEAT'),
                'excluded_before_mapping':raw_counts[symbol]-kept_counts.get(symbol,0)})
    save_csv(class_rows, args.output_dir/'class_distribution.csv')
    save_csv(symbol_rows, args.output_dir/'original_symbol_counts.csv')

    fig, axes = plt.subplots(3,2,figsize=(15,11))
    for row, rid in enumerate(ids):
        data = summary['records'][rid]
        for column, key in enumerate(('original_all_annotation_symbol_counts','original_segment_symbol_counts')):
            counts = data[key]
            axes[row,column].bar(list(counts),list(counts.values()))
            axes[row,column].set_title(f'Record {rid}: '+('all .atr annotations' if column==0 else 'complete heartbeat segments'))
            axes[row,column].set_xlabel('Original WFDB symbol (case-sensitive)')
            axes[row,column].set_ylabel('Count')
    fig.tight_layout()
    try:
        save_comparison_figure(fig,args.output_dir/'original_symbols.png')
    finally:
        plt.close(fig)

    fig, axes = plt.subplots(2,2,figsize=(13,9))
    for ax, (rid, data) in zip(axes.flat,[*summary['records'].items(),('ALL_SELECTED',summary['aggregate'])]):
        counts = [data['class_counts'][name] for name in CLASS_NAMES]
        bars = ax.bar(CLASS_NAMES,counts)
        ax.bar_label(bars,padding=3)
        ax.set_title(f'{rid}: AAMI-style mapped beats; unmapped={data["unmapped_count"]}')
        ax.set_ylabel('Count (unbalanced)')
        ax.set_ylim(0,max(counts,default=0)*1.15+1)
    fig.tight_layout()
    try:
        save_comparison_figure(fig,args.output_dir/'aami_class_counts.png')
    finally:
        plt.close(fig)

    # Retain at most one small example per class, rather than all records' arrays.
    examples = {}
    for rid in ids:
        beats = map_segments(segment_recording(load_recording(rid,args.data_dir)),unknown_policy='mark')
        for name in CLASS_NAMES:
            matches = np.flatnonzero(beats.y==CLASS_INDICES[name])
            if name not in examples and len(matches):
                i = int(matches[0])
                examples[name] = {'signal':beats.X[i].copy(), 'record_id':rid,
                    'position':int(beats.segments.annotation_positions[i]),
                    'symbol':str(beats.segments.annotation_symbols[i]),
                    'channel_names':beats.segments.channel_names,
                    'fs':beats.segments.sampling_frequency,'before':beats.segments.before_samples}
    present = [name for name in CLASS_NAMES if name in examples]
    absent = [name for name in CLASS_NAMES if name not in examples]
    fig, axes = plt.subplots(2,len(present),squeeze=False,figsize=(4.5*len(present),7))
    for column, name in enumerate(present):
        example = examples[name]
        time = (np.arange(example['signal'].shape[1])-example['before'])/example['fs']
        for channel in range(2):
            ax = axes[channel,column]
            ax.plot(time,example['signal'][channel])
            ax.axvline(0,color='tab:red',linestyle='--',linewidth=0.8)
            ax.set_title(f'{name} ({CLASS_INDICES[name]}) <- {example["symbol"]}\nRecord {example["record_id"]}, sample {example["position"]}')
            ax.set_xlabel('Time relative to expert annotation (s)')
            ax.set_ylabel(f'{example["channel_names"][channel]} (dimensionless)')
            ax.grid(alpha=0.25)
    fig.suptitle('Actual filtered + Z-score segments | classes absent in selected records: '+(', '.join(absent) or 'none'))
    fig.tight_layout()
    try:
        save_comparison_figure(fig,args.output_dir/'available_class_examples.png')
    finally:
        plt.close(fig)
    summary['example_beats'] = {name:{k:v for k,v in example.items() if k not in ('signal','channel_names')}
                                for name,example in examples.items()}
    summary['absent_example_classes'] = absent
    with (args.output_dir/'label_mapping_summary.json').open('x') as output:
        json.dump(summary,output,indent=2,allow_nan=False)
    for rid, data in summary['records'].items():
        print(rid, data['class_counts'], 'segments:',data['segment_count'],'unmapped:',data['unmapped_symbol_counts'])
    print('Saved plots, CSV tables and diagnostic JSON to',args.output_dir)


if __name__=='__main__':
    main()
