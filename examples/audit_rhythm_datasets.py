"""Audit annotations/metadata only; existing heartbeat ECG files are never rewritten."""
import argparse
import csv
import json
from pathlib import Path
from src.rhythm_audit import audit_mitdb, summarize_ptb_metadata, rhythm_duration_capacity
from src.prepare_dataset import sha256_file


def write_json(path,data):
    with path.open('x') as out:
        json.dump(data,out,indent=2,sort_keys=True,allow_nan=False)


def write_csv(path,rows):
    if not rows:
        return
    with path.open('x',newline='',encoding='utf-8') as out:
        writer=csv.DictWriter(out,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows([{k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('results/preprocessing/phase11_rhythm_audit'))
    parser.add_argument('--ptb-database',type=Path)
    parser.add_argument('--scp-statements',type=Path)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    result=audit_mitdb()
    write_json(args.output_dir/'mitdb_rhythm_audit.json',result)
    write_csv(args.output_dir/'mitdb_rhythm_inventory.csv',result['inventory'])
    write_csv(args.output_dir/'mitdb_theoretical_10s_capacity.csv',rhythm_duration_capacity(result))
    for kind in ('intervals','episodes','markers','transitions'):
        rows=[row for record in result['records'].values() for row in record[kind]]
        write_csv(args.output_dir/f'mitdb_rhythm_{kind}.csv',rows)
    write_csv(args.output_dir/'mitdb_unknown_intervals.csv',result['unknown_intervals'])
    if (args.ptb_database is None)!=(args.scp_statements is None):
        raise ValueError('Supply both local PTB metadata CSV paths, or neither')
    if args.ptb_database is not None:
        ptb=summarize_ptb_metadata(args.ptb_database,args.scp_statements)
        ptb['evidence']='Actual supplied CSV metadata, not waveform audit; verify completeness and version from provenance'
        ptb['input_sha256']={str(p):sha256_file(p) for p in (args.ptb_database,args.scp_statements)}
    else:
        ptb={'evidence':'Official documentation and browser definitions verified; full local record metadata unavailable',
             'full_metadata_class_counts':None,'full_metadata_patient_counts':None,
             'reason':'No local CSV found; shell DNS unavailable; official browser database preview is partial and cannot support full class counts',
             'source':'https://physionet.org/content/ptb-xl/1.0.3/'}
    write_json(args.output_dir/'ptbxl_metadata_audit.json',ptb)
    for row in result['inventory']:
        print(row['rhythm_code'],row['record_count'],'records',row['patient_group_count'],'groups',
              row['observed_episodes'],'observed episodes',round(row['duration_seconds'],6),'seconds')
    print('Unknown duration:',result['unknown_duration_seconds'],'seconds',result['unknown_reason_counts'])
    print('Confirmed state changes:',result['confirmed_label_changes'])


if __name__=='__main__':
    main()
