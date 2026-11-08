"""Audit-only annotation intervals and SCP metadata; no waveform preprocessing."""
import ast
from collections import Counter, defaultdict
from itertools import groupby
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
import wfdb
from .data_loader import DEFAULT_DATA_DIR, list_recordings, locate_data_dir

# Source: official PhysioBank rhythm annotation dictionary. No AAMI labels here.
MIT_RHYTHMS = {
    'AB':'Atrial bigeminy','AFIB':'Atrial fibrillation','AFL':'Atrial flutter',
    'B':'Ventricular bigeminy','BII':'Second-degree heart block','IVR':'Idioventricular rhythm',
    'N':'Normal sinus rhythm','NOD':'Nodal (A-V junctional) rhythm','P':'Paced rhythm',
    'PREX':'Pre-excitation (WPW)','SBR':'Sinus bradycardia','SVTA':'Supraventricular tachyarrhythmia',
    'T':'Ventricular trigeminy','VFL':'Ventricular flutter','VT':'Ventricular tachycardia'}


def parse_rhythm_note(note):
    """Parse a complete + auxiliary tag, stripping only padding/optional final parenthesis.

    Unknown tags remain unknown; no rate thresholds or heartbeat labels infer rhythm.
    """
    if not isinstance(note,str):
        raise ValueError('Auxiliary notes must be strings')
    cleaned=note.strip(' \t\r\n\x00')
    match=re.fullmatch(r'\(([A-Za-z0-9]+)\)?',cleaned)
    if match is None:
        return None,'malformed_or_empty_rhythm_marker'
    code=match.group(1)
    return (code,'documented_rhythm') if code in MIT_RHYTHMS else (None,'undocumented_rhythm_tag')


def rhythm_intervals(samples, symbols, notes, *, sampling_frequency, sample_count, record_id):
    """Partition a recording into half-open state intervals at expert '+' positions.

    State persists until next + or EOF (an explicit inference rule, not proof of
    clinical onset/end). Initial unlabeled samples are UNKNOWN. Conflicting markers
    at a shared position make the following interval ambiguous, never last-wins.
    Adjacent same-state intervals are retained as boundaries but merged as episodes.
    """
    fs=float(sampling_frequency)
    if not np.isfinite(fs) or fs<=0 or isinstance(sample_count,bool) or not isinstance(sample_count,(int,np.integer)) or sample_count<1:
        raise ValueError('Need positive finite fs and a positive integer sample_count')
    positions=np.asarray(samples)
    if positions.ndim!=1 or (positions.size and positions.dtype.kind not in 'iu'):
        raise ValueError('Annotation positions must be 1D integers')
    if len(positions)!=len(symbols) or len(symbols)!=len(notes):
        raise ValueError('Annotation arrays must have matching lengths')
    if np.any(positions<0) or np.any(positions>=sample_count):
        raise ValueError('Annotation outside recording')
    positions=positions.astype(np.int64)
    if np.any(np.diff(positions)<0):
        raise ValueError('Annotations must be ordered')
    markers=[]
    ignored=[]
    for i,(pos,symbol,note) in enumerate(zip(positions,symbols,notes)):
        if not isinstance(note,str):
            raise ValueError('Auxiliary notes must be strings')
        if symbol!='+':
            if note.strip(' \x00').startswith('('):
                ignored.append({'annotation_index':i,'sample':int(pos),'symbol':symbol,'note':note})
            continue
        label,status=parse_rhythm_note(note)
        markers.append({'record_id':record_id,'annotation_index':i,'sample':int(pos),
                        'raw_aux_note':note,'label':label,'status':status})
    grouped=[]
    for pos,group in groupby(markers,key=lambda row:row['sample']):
        group=list(group)
        labels={r['label'] for r in group}
        label=next(iter(labels)) if len(labels)==1 else None
        if len(group)>1 and (len(labels)>1 or label is None):
            status='conflicting_same_sample_markers'
        elif label is None:
            status=group[0]['status']
        else:
            status='documented_rhythm'
        grouped.append({'sample':pos,'label':label,'status':status,
                        'source_annotation_indices':[r['annotation_index'] for r in group]})
    intervals=[]
    if not grouped or grouped[0]['sample']>0:
        intervals.append({'record_id':record_id,'start_sample':0,
            'stop_sample':grouped[0]['sample'] if grouped else int(sample_count),
            'label':None,'status':'unlabeled_prefix' if grouped else 'no_rhythm_annotations',
            'source_annotation_indices':[],'initial_state':True,'right_censored':not grouped})
    for i,row in enumerate(grouped):
        stop=grouped[i+1]['sample'] if i+1<len(grouped) else int(sample_count)
        intervals.append({'record_id':record_id,'start_sample':row['sample'],'stop_sample':stop,
            'label':row['label'],'status':row['status'],'source_annotation_indices':row['source_annotation_indices'],
            'initial_state':i==0,'right_censored':i==len(grouped)-1})
    episodes=[]
    for row in intervals:
        row.update({'duration_samples':row['stop_sample']-row['start_sample'],
                    'start_seconds':row['start_sample']/fs,'stop_seconds':row['stop_sample']/fs,
                    'duration_seconds':(row['stop_sample']-row['start_sample'])/fs})
        if episodes and (episodes[-1]['label'],episodes[-1]['status'])==(row['label'],row['status']):
            episodes[-1]['stop_sample']=row['stop_sample']
            episodes[-1]['duration_samples']+=row['duration_samples']
            episodes[-1]['duration_seconds']+=row['duration_seconds']
            episodes[-1]['stop_seconds']=row['stop_seconds']
            episodes[-1]['right_censored']=row['right_censored']
            episodes[-1]['source_annotation_indices']+=row['source_annotation_indices']
        else:
            episodes.append({**row,'source_annotation_indices':list(row['source_annotation_indices'])})
    transitions=[{'record_id':record_id,'sample':b['start_sample'],'seconds':b['start_sample']/fs,
                  'from_label':a['label'],'to_label':b['label'],'from_status':a['status'],'to_status':b['status'],
                  'confirmed_label_change':a['label'] is not None and b['label'] is not None and a['label']!=b['label']}
                 for a,b in zip(episodes,episodes[1:])]
    if sum(r['duration_samples'] for r in intervals)!=sample_count:
        raise ValueError('Intervals do not partition the complete recording')
    return {'markers':markers,'intervals':intervals,'episodes':episodes,'transitions':transitions,
            'ignored_rhythm_like_nonplus_notes':ignored}


def audit_mitdb(root=DEFAULT_DATA_DIR):
    """Read .hea/.atr only. Preserve WFDB source coordinates and recorded unknowns."""
    directory=locate_data_dir(root)
    records={}
    with (Path(__file__).resolve().parents[1]/'metadata/patient_groups.json').open() as source:
        groups=json.load(source)['recordings']
    for rid in list_recordings(directory):
        header=wfdb.rdheader(str(directory/rid))
        annotation=wfdb.rdann(str(directory/rid),'atr')
        result=rhythm_intervals(annotation.sample,annotation.symbol,annotation.aux_note,
            sampling_frequency=header.fs,sample_count=header.sig_len,record_id=rid)
        result.update({'sampling_frequency':header.fs,'sample_count':header.sig_len,
                       'channel_names':header.sig_name,'units':header.units,
                       'patient_group':groups[rid]['patient_group']})
        records[rid]=result
    inventory=[]
    for code,description in MIT_RHYTHMS.items():
        found=[rid for rid,result in records.items() if any(e['label']==code for e in result['episodes'])]
        episodes=[e for result in records.values() for e in result['episodes'] if e['label']==code]
        inventory.append({'rhythm_code':code,'description':description,'record_count':len(found),'record_ids':found,
            'patient_group_count':len({records[rid]['patient_group'] for rid in found}),
            'observed_episodes':len(episodes),'duration_seconds':sum(e['duration_seconds'] for e in episodes),
            'right_censored_episodes':sum(e['right_censored'] for e in episodes),
            'initial_state_episodes':sum(e['initial_state'] for e in episodes)})
    unknown=[r for result in records.values() for r in result['intervals'] if r['label'] is None]
    return {'records':records,'inventory':inventory,'record_count':len(records),
            'total_duration_seconds':sum(r['sample_count']/r['sampling_frequency'] for r in records.values()),
            'unknown_duration_seconds':sum(r['duration_seconds'] for r in unknown),
            'unknown_intervals':unknown,'unknown_reason_counts':dict(Counter(r['status'] for r in unknown)),
            'episode_definition':'Maximal adjacent same documented state; observed runs, not verified clinical episode onsets/ends',
            'interval_rule':'Half-open expert + state to next +, final state carried to EOF with right censoring',
            'confirmed_label_changes':sum(t['confirmed_label_change'] for r in records.values() for t in r['transitions'])}


def parse_scp_codes(value):
    """Safely parse recorded SCP likelihood dictionaries; 0 means unknown, not absent."""
    try:
        data=ast.literal_eval(value) if isinstance(value,str) else value
    except (ValueError,SyntaxError) as exc:
        raise ValueError('Invalid scp_codes literal') from exc
    if not isinstance(data,dict):
        raise ValueError('scp_codes must be a dictionary')
    for key,score in data.items():
        if not isinstance(key,str) or not key or isinstance(score,bool) or not isinstance(score,(int,float)) or not np.isfinite(score) or not 0<=score<=100:
            raise ValueError('Invalid SCP code or likelihood')
    return dict(data)


def summarize_ptb_metadata(database_path, statements_path):
    """Only for an actual local metadata CSV; counts presence, including score 0.

    This does not load waveforms or create target arrays/splits. Caller must verify
    file completeness/version and preserve its provenance before claiming full counts.
    """
    table=pd.read_csv(database_path)
    definitions=pd.read_csv(statements_path,index_col=0)
    required={'ecg_id','patient_id','scp_codes','strat_fold'}
    if not required.issubset(table.columns) or 'rhythm' not in definitions.columns:
        raise ValueError('Required metadata columns missing')
    if table['ecg_id'].duplicated().any() or table['patient_id'].isna().any():
        raise ValueError('Duplicate ECG IDs or missing patient IDs')
    codes=[parse_scp_codes(s) for s in table['scp_codes']]
    rhythms=definitions.index[definitions['rhythm']==1].tolist()
    result=[]
    for code in rhythms:
        rows=[i for i,c in enumerate(codes) if code in c]
        scores=Counter(str(codes[i][code]) for i in rows)
        result.append({'code':code,'description':definitions.loc[code,'description'],
            'record_count':len(rows),'patient_count':int(table.iloc[rows]['patient_id'].nunique()),
            'likelihood_value_counts':dict(scores)})
    multilabel=sum(sum(code in row for code in rhythms)>1 for row in codes)
    by_patient=table.groupby('patient_id')['strat_fold'].nunique(dropna=False)
    if table['strat_fold'].isna().any() or not table['strat_fold'].isin(range(1,11)).all():
        raise ValueError('Invalid recommended fold metadata')
    return {'record_count':len(table),'patient_count':int(table['patient_id'].nunique()),'rhythm_counts':result,
        'multiple_rhythm_code_records':multilabel,'patients_across_multiple_folds':int((by_patient>1).sum()),
        'count_rule':'presence of dictionary key regardless of likelihood; no waveform localization implied'}


def rhythm_duration_capacity(audit, duration_seconds=10.0):
    """Theoretical nonoverlapping interval capacity only; creates NO waveform windows."""
    if not np.isfinite(duration_seconds) or duration_seconds<=0:
        raise ValueError('Positive finite duration required')
    rows=[]
    for code in MIT_RHYTHMS:
        record_ids=[]
        slots=long_count=short_count=0
        for rid,record in audit['records'].items():
            length=int(round(duration_seconds*record['sampling_frequency']))
            if length<1:
                raise ValueError('Duration rounds to zero samples')
            present=False
            for episode in record['episodes']:
                if episode['label']!=code:
                    continue
                capacity=episode['duration_samples']//length
                slots+=capacity
                if capacity:
                    long_count+=1
                    present=True
                else:
                    short_count+=1
            if present:
                record_ids.append(rid)
        rows.append({'rhythm_code':code,'window_seconds':duration_seconds,
            'theoretical_nonoverlapping_slots':slots,'episodes_at_least_window_length':long_count,
            'shorter_episodes':short_count,'qualifying_record_ids':record_ids,
            'qualifying_patient_groups':len({audit['records'][rid]['patient_group'] for rid in record_ids}),
            'interpretation':'Arithmetic upper capacity in carried-forward annotation states; no edge margin, no extracted ECG windows'})
    return rows
