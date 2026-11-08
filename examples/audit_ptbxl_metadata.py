"""Phase 12: analyze local CSV metadata only, without changing sources or splits."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.ptbxl_metadata import (DEFAULT_METADATA_DIR,CORE_TARGETS,QUALITY_COLUMNS,
    load_ptb_metadata,audit_metadata,propose_exclusive_cohort,MetadataValidationError)


def write_json(path,data):
    with Path(path).open('x',encoding='utf-8') as output:
        json.dump(data,output,indent=2,sort_keys=True,allow_nan=False)


def write_csv(path,rows):
    with Path(path).open('x',newline='',encoding='utf-8') as output:
        writer=csv.DictWriter(output,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _json_value(value):
    if pd.isna(value):
        return None
    return value.item() if isinstance(value,np.generic) else value


def run_audit(metadata_dir=DEFAULT_METADATA_DIR,output_dir=Path('results/preprocessing/phase12_ptbxl_metadata_audit'),report_dir=Path('.')):
    """Write new analysis artifacts. Existing outputs are never overwritten."""
    output_dir=Path(output_dir)
    report_dir=Path(report_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    report_dir.mkdir(parents=True,exist_ok=True)
    try:
        data=load_ptb_metadata(metadata_dir)
    except (MetadataValidationError,FileNotFoundError) as exc:
        write_json(output_dir/'validation_failure.json',{'error':str(exc),'counts_generated':False})
        raise
    summary=audit_metadata(data)
    # Counterfactual cohort counts only; no records or waveforms are filtered out.
    for name,targets in [('core_plus_aflt',CORE_TARGETS+('AFLT',)),('core_plus_sarrh',CORE_TARGETS+('SARRH',))]:
        proposal=propose_exclusive_cohort(data,targets)
        summary.setdefault('alternative_proposals',{})[name]={k:v for k,v in proposal.items() if k!='record_audit'}
    record_rows=[]
    patient_rows=[]
    fold_rows=[]
    for row in summary['rhythm_counts']:
        code=row['code']
        proposed=summary['proposal']['per_class'][code]
        base={'code':code,'description':row['description']}
        record_rows.append({**base,'record_count':row['record_count'],'percent_all_records':row['record_percent'],
            'zero_likelihood_records':row['zero_likelihood_records'],
            **{f'fold_{f}_records':row['records_by_fold'][str(f)] for f in range(1,11)},
            'proposed_records_after':proposed['records_after']})
        patient_rows.append({**base,'distinct_patients':row['patient_count'],
            **{f'fold_{f}_patients':row['patients_by_fold'][str(f)] for f in range(1,11)},
            'proposed_patients_after':proposed['patients_after']})
        for fold in range(1,11):
            key=str(fold)
            fold_rows.append({'code':code,'strat_fold':fold,
                'records_before':row['records_by_fold'][key],'patients_before':row['patients_by_fold'][key],
                'records_after_proposal':proposed['records_after_by_fold'][key],
                'patients_after_proposal':proposed['patients_after_by_fold'][key]})
    write_csv(report_dir/'PTBXL_RHYTHM_CLASS_COUNTS.csv',record_rows)
    write_csv(report_dir/'PTBXL_RHYTHM_PATIENT_COUNTS.csv',patient_rows)
    write_csv(report_dir/'PTBXL_LABEL_COOCCURRENCE.csv',summary['cooccurrence'])
    write_csv(output_dir/'rhythm_counts_by_fold.csv',fold_rows)
    write_csv(output_dir/'statement_definitions.csv',summary['definitions'])
    proposal_rows=[{**row,'original_rhythm_codes':json.dumps(row['original_rhythm_codes'])}
                   for row in summary['proposal']['record_audit']]
    write_csv(output_dir/'proposed_cohort_audit.csv',proposal_rows)
    write_csv(output_dir/'missing_values.csv',[{'column':name,'missing_values':count} for name,count in summary['missing_values_by_column'].items()])
    with (output_dir/'original_statements_and_quality.jsonl').open('x',encoding='utf-8') as output:
        for index,(_,row) in enumerate(data.records.iterrows()):
            item={'ecg_id':int(data.ecg_ids[index]),'patient_id':int(data.patient_ids[index]),'strat_fold':int(data.folds[index]),
                'filename_lr':row['filename_lr'],'filename_hr':row['filename_hr'],
                'original_scp_codes_cell':row['scp_codes'],'original_scp_statements':data.statements[index],
                'quality_metadata':{name:_json_value(row[name]) for name in QUALITY_COLUMNS if name in row}}
            output.write(json.dumps(item,sort_keys=True,allow_nan=False)+'\n')
    summary['proposal'].pop('record_audit')  # A separate CSV preserves EVERY row and its prospective reason.
    write_json(output_dir/'ptbxl_metadata_audit.json',summary)
    for row in summary['rhythm_counts']:
        print(row['code'],row['record_count'],'records;',row['patient_count'],'patients;',
              'likelihood',row['likelihood_counts'],flush=True)
    print('Total:',summary['record_count'],'records;',summary['patient_count'],'patients',flush=True)
    print('Multiple rhythms:',summary['records_with_multiple_rhythms'],'No rhythm:',summary['records_without_rhythm'],flush=True)
    print('Patient fold conflicts:',summary['patient_fold_conflict_count'],flush=True)
    print('Proposed four-class cohort:',summary['proposal']['proposed_record_count'],summary['proposal']['proposed_patient_count'],flush=True)
    print('Proposal status counts:',summary['proposal']['mutually_exclusive_exclusion_reasons'],flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata-dir',type=Path,default=DEFAULT_METADATA_DIR)
    parser.add_argument('--output-dir',type=Path,default=Path('results/preprocessing/phase12_ptbxl_metadata_audit'))
    parser.add_argument('--report-dir',type=Path,default=Path('.'))
    args=parser.parse_args()
    run_audit(args.metadata_dir,args.output_dir,args.report_dir)


if __name__=='__main__':
    main()
