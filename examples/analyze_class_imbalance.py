"""Analyze every complete local MIT-BIH recording, without balancing any ECG."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from src.data_loader import DEFAULT_DATA_DIR
from src.label_mapping import CLASS_NAMES
from src.class_imbalance import analyze_database
from src.preprocessing_comparison import save_comparison_figure
import matplotlib.pyplot as plt


def save_csv(rows,path):
    with path.open('x',newline='',encoding='utf-8') as output:
        writer=csv.DictWriter(output,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=DEFAULT_DATA_DIR)
    parser.add_argument('--output-dir',type=Path,default=Path('results/preprocessing/phase9'))
    parser.add_argument('--rare-count-threshold',type=int,default=1000)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    summary=analyze_database(args.data_dir,rare_count_threshold=args.rare_count_threshold,
        progress=lambda rid,r:print(f'{rid}: {r["mapped_count"]} mapped / {r["segment_count"]} segments',flush=True))
    total=summary['aggregate']
    rows=[{'class_name':name,'class_index':i,'count':total['class_counts'][name],
           'percent_of_mapped':total['class_percentages'][name],
           'is_zero':name in total['zero_classes'],'is_rare':name in total['rare_classes']}
          for i,name in enumerate(CLASS_NAMES)]
    save_csv(rows,args.output_dir/'class_distribution.csv')
    per_record=[]
    for rid,r in summary['records'].items():
        row={'record_id':rid,'segment_count':r['segment_count'],'mapped_count':r['mapped_count'],
             'unmapped_count':r['unmapped_count'],'unresolved_patient_groups':r['patient_groups_unresolved_count']}
        row.update({f'{name}_count':r['class_counts'][name] for name in CLASS_NAMES})
        row.update({f'{name}_percent':r['class_percentages'][name] for name in CLASS_NAMES})
        row.update({'rare_classes':';'.join(r['rare_classes']),'missing_classes':';'.join(r['zero_classes']),
            'majority_classes':';'.join(r['majority_classes']),
            'minority_nonempty_classes':';'.join(r['minority_nonempty_classes']),
            'nonempty_ratio':r['majority_to_minority_nonempty_ratio']})
        per_record.append(row)
    save_csv(per_record,args.output_dir/'recording_class_distribution.csv')
    save_csv([{'record_id':rid,'patient_group_id':p['patient_group_id'] or '',
               'patient_group_status':'unresolved' if p['patient_group_id'] is None else 'provided'}
              for rid,r in summary['records'].items() for p in r['record_patient_groups']],
             args.output_dir/'recording_patient_groups.csv')
    audit=[]
    for rid,r in summary['records'].items():
        s=r['segmentation_report']
        audit.append({'record_id':rid,'all_annotations':s['total_annotations'],
            'complete_segments':r['segment_count'],'mapped':r['mapped_count'],'unmapped':r['unmapped_count'],
            'excluded_nonheartbeat':s['excluded_nonheartbeat'],'skipped_boundary':s['skipped_boundary'],
            'unmapped_symbols':json.dumps(r['unmapped_symbol_counts'],sort_keys=True),
            'excluded_nonheartbeat_symbols':json.dumps(s['excluded_symbol_counts'],sort_keys=True)})
    save_csv(audit,args.output_dir/'segmentation_audit.csv')
    counts=[total['class_counts'][name] for name in CLASS_NAMES]
    percentages=[total['class_percentages'][name] for name in CLASS_NAMES]
    for filename,values,title,ylabel,logarithmic in [
        ('class_counts.png',counts,'All local MIT-BIH: mapped heartbeat counts','Count',False),
        ('class_percentages.png',percentages,'All local MIT-BIH: class percentages','Percent of mapped segments',False),
        ('class_counts_log.png',counts,'All local MIT-BIH: nonempty class counts (log scale)','Count (log scale)',True)]:
        fig,ax=plt.subplots(figsize=(10,6))
        shown=[i for i,v in enumerate(values) if v>0] if logarithmic else list(range(5))
        bars=ax.bar([CLASS_NAMES[i] for i in shown],[values[i] for i in shown])
        ax.bar_label(bars,fmt='%.3f' if filename=='class_percentages.png' else '%g',padding=3)
        if logarithmic:
            ax.set_yscale('log')
            ax.set_ylim(bottom=0.5,top=max(counts,default=1)*2)
        else:
            ax.set_ylim(0,max(values,default=1)*1.15+1)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        fig.tight_layout()
        try:
            save_comparison_figure(fig,args.output_dir/filename)
        finally:
            plt.close(fig)
    matrix=np.array([[r['class_counts'][name] for name in CLASS_NAMES] for r in summary['records'].values()])
    fig,ax=plt.subplots(figsize=(10,16))
    # log1p keeps genuine zeros visible; annotations give actual counts.
    image=ax.imshow(np.log1p(matrix),aspect='auto',cmap='viridis')
    ax.set_xticks(range(5),CLASS_NAMES)
    ax.set_yticks(range(len(per_record)),summary['record_ids'])
    ax.set_xlabel('AAMI-style class')
    ax.set_ylabel('Recording ID (not patient ID)')
    ax.set_title('Recording-by-class counts (color = ln(1 + count))')
    for row in range(len(matrix)):
        for col in range(5):
            ax.text(col,row,str(matrix[row,col]),ha='center',va='center',fontsize=7,
                    color='black' if np.log1p(matrix[row,col])>6 else 'white')
    fig.colorbar(image,ax=ax,label='ln(1 + count)')
    fig.tight_layout()
    try:
        save_comparison_figure(fig,args.output_dir/'recording_class_heatmap.png')
    finally:
        plt.close(fig)
    with (args.output_dir/'imbalance_summary.json').open('x') as output:
        json.dump(summary,output,indent=2,allow_nan=False)
    print('Aggregate:',json.dumps(total,indent=2),flush=True)
    print('No balancing applied. Outputs:',args.output_dir,flush=True)


if __name__=='__main__':
    main()
