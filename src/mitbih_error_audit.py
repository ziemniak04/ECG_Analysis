"""Read-only Phase-23B checks, deterministic sample selection and ECG figures.

Never construct final-test data, fit a model, alter preprocessing, or write inputs.
Saved validation predictions are authoritative; only record 232 has fresh TRAIN
inference. Reconstruction checks reuse existing filters/frozen normalization in RAM.
"""
from collections import Counter
import csv
import json
from pathlib import Path
import numpy as np
import wfdb
from .label_mapping import DEFAULT_MAPPING, CLASS_NAMES, CLASS_INDICES
from .visualization import BEAT_SYMBOLS
from .mitbih_protocol import MITBIHExportIndex,load_split
from .mitbih_readiness import verify_frozen_protocol
from .prepare_dataset import sha256_file
from .data_loader import DEFAULT_DATA_DIR,load_recording
from .preprocessing import filter_recording,ECGNormalizer
from .mitbih_full_run import csv_new
from .ptbxl_training import write_json_new

COMPARISON=Path('results/model_training/phase23a_mitbih_unweighted/run_001/comparison')
MODEL_PATHS={
 'A':Path('results/model_training/phase22_mitbih_full/run_001/best_model.pt'),
 'B':Path('results/model_training/phase23a_mitbih_unweighted/run_001/best_model.pt')}
MODEL_HASHES={'A':'83f914122bf389fe0ddde1abe229cafa81abf1e8f8973dbd03cfc86eacb00d75',
 'B':'6f1b46426fa5202ae7d958aafd1ea0bf918bf11be0eb41cc0cc53230cadbc4af'}


def require_record(index,split,rid):
    """Refuse test IDs BEFORE raw header/annotation/NPZ/file hashing."""
    for part in ('train','validation'):
        if index.by_record[rid].patient_group in split['groups'][part]:return part
    raise PermissionError('Final MIT-BIH test records are forbidden')


def predictions(index,split):
    out={}
    for model,name in [('A','A_sqrt_balanced'),('B','B_unweighted')]:
        path=COMPARISON/(name+'_validation_predictions.csv')
        sealed=json.loads((COMPARISON/'ARTIFACT_CHECKSUMS.json').read_text())
        if sha256_file(path)!=sealed[str(path)]:raise ValueError('Saved predictions changed')
        with path.open() as f:rows=list(csv.DictReader(f))
        for row in rows:
            if require_record(index,split,row['record_id'])!='validation':raise ValueError('Non-validation prediction')
            row['source_index']=int(row['source_index']);row['annotation_position']=int(row['annotation_position'])
            probs=np.array([float(row['p_'+c]) for c in CLASS_NAMES])
            if not np.isfinite(probs).all() or np.any(probs<0) or not np.isclose(probs.sum(),1,atol=1e-6):
                raise ValueError('Invalid stored Softmax scores')
            if CLASS_NAMES[int(probs.argmax())]!=row['predicted_class']:raise ValueError('Argmax differs')
            row['score']=float(probs.max());row['partition']='validation'
        out[model]={r['source_index']:r for r in rows}
        if len(out[model])!=23134:raise ValueError('Incomplete or duplicate validation')
    if set(out['A'])!=set(out['B']):raise ValueError('Different source populations')
    for i,a in out['A'].items():
        b=out['B'][i]
        for key in ('record_id','patient_group','annotation_position','original_symbol','true_class','channel_0','channel_1'):
            if a[key]!=b[key]:raise ValueError('A/B metadata disagreement')
    return out


def verify_annotations(arrays,annotation,header,config):
    """Check every exported beat versus original ATR indices and extraction bounds."""
    positions=np.asarray(annotation.sample,dtype=np.int64);symbols=np.array(annotation.symbol)
    ai=arrays['annotation_indices'];ap=arrays['annotation_positions']
    if np.any(ai<0) or np.any(ai>=len(positions)) or not np.array_equal(positions[ai],ap):
        raise ValueError('Original annotation position/index mismatch')
    if not np.array_equal(symbols[ai],arrays['original_symbols']):raise ValueError('Original symbols mismatch')
    if tuple(header.sig_name)!=tuple(arrays['channel_names']):raise ValueError('Channel order differs from raw header')
    before,after=config['before_samples'],config['after_samples']
    if (before,after)!=(90,162) or header.fs!=360 or arrays['X'].shape[1:]!=(2,252):
        raise ValueError('Window/fs contract differs')
    valid=[i for i,(p,s) in enumerate(zip(positions,symbols)) if s in BEAT_SYMBOLS and p>=before and p+after<=header.sig_len]
    if not np.array_equal(ai,np.asarray(valid,dtype=np.int64)):raise ValueError('Unexpected dropped/reordered heartbeat')
    if any(DEFAULT_MAPPING.get(str(s))!=CLASS_NAMES[int(y)] for s,y in zip(arrays['original_symbols'],arrays['y'])):
        raise ValueError('Stored AAMI labels disagree')
    if config['normalization']['scope']!='recording' or config['normalization']['sample_count']!=header.sig_len:
        raise ValueError('Normalization provenance differs')
    f=config['filtering']
    if f!={'highpass_hz':.5,'lowpass_hz':45.,'notch_hz':None,'order':4,'quality_factor':30.}:
        raise ValueError('Unexpected filtering settings')
    if (config['normalization']['method']!='zscore' or config['normalization']['channel_names']!=list(header.sig_name)
        or config['normalization']['source_ids']!=[str(arrays['record_ids'][0])]):raise ValueError('Normalization source/lead differs')
    overlap_pairs=0;overlap_values=0
    for i,gap in enumerate(np.diff(ap)):
        gap=int(gap)
        if 0<gap<252:
            a,b=arrays['X'][i,:,gap:],arrays['X'][i+1,:,:252-gap]
            if not np.array_equal(a,b):raise ValueError('Overlapping segments do not share the same signal samples')
            overlap_pairs+=1;overlap_values+=a.size
    skipped=[int(p) for p,s in zip(positions,symbols) if s in BEAT_SYMBOLS and (p<before or p+after>header.sig_len)]
    return dict(exported_beats=len(ai),raw_annotations=len(positions),skipped_boundary=len(skipped),
        skipped_boundary_positions=skipped,nonbeat_counts=dict(Counter(s for s in symbols if s not in BEAT_SYMBOLS)),
        overlap_pairs=overlap_pairs,identical_overlap_values=overlap_values,
        edge_tag_beats_5seconds=int(np.sum((ap-before<1800)|(header.sig_len-(ap+after)<1800))),
        min_left_margin_samples=int(np.min(ap-before)),min_right_margin_samples=int(np.min(header.sig_len-(ap+after))))


def reconstruct(record,arrays,config):
    """In-memory equivalence check, not a new dataset; never refit normalization."""
    filtered=filter_recording(record,**config['filtering'])
    normalizer=ECGNormalizer.from_dict(config['normalization'])
    normalized=normalizer.transform(filtered.signals,channel_names=record.channel_names)
    max_difference=0.
    for i,p in enumerate(arrays['annotation_positions']):
        expected=normalized[p-90:p+162].T.astype(np.float32)
        delta=float(np.max(np.abs(arrays['X'][i]-expected)));max_difference=max(max_difference,delta)
        if not np.array_equal(arrays['X'][i],expected):raise ValueError('Frozen pipeline reconstruction differs')
    return dict(beats_checked=len(arrays['X']),float32_bitwise_equal=True,max_abs_difference=max_difference,
        filtered_offset_difference=float(np.max(np.abs(filtered.signals.mean(axis=0)-np.array(config['normalization']['offset'])))),
        filtered_scale_difference=float(np.max(np.abs(filtered.signals.std(axis=0)-np.array(config['normalization']['scale'])))),
        normalized_full_record_mean=normalized.mean(axis=0).tolist(),normalized_full_record_std=normalized.std(axis=0).tolist())


def mapping_rows(raw_counts,export_counts):
    from wfdb.io.annotation import ann_label_table
    definitions={str(r.symbol):str(r.description) for r in ann_label_table.itertuples()}
    symbols=sorted(set(definitions)|set(raw_counts['train'])|set(raw_counts['validation'])|set(DEFAULT_MAPPING))
    result=[]
    for symbol in symbols:
        group=DEFAULT_MAPPING.get(symbol,'')
        if group:policy='mapped_by_documented_project_protocol'
        elif symbol in BEAT_SYMBOLS:policy='recognized_beat_unmapped_error_if_exported'
        elif symbol=='?':policy='learning_beat_excluded_by_project_beat_set'
        else:policy='nonbeat_or_not_assigned_excluded'
        note={'a':'Aberrated atrial premature; remains S, not V',
              'f':'Paced-normal fusion; Q, distinct from F',
              'F':'Ventricular-normal fusion; F, distinct from f',
              '/':'Paced beat; grouped Q by this protocol, not a pacing artifact',
              '^':'Noncaptured pacing artifact; not a paced heartbeat',
              'Q':'Literal unclassifiable beat; only one subtype of grouped Q',
              'x':'Blocked P annotation is non-beat here; not silently S',
              'N':'Group N also includes L/R/e/j, not a healthy/sinus diagnosis'}.get(symbol,'')
        result.append(dict(symbol=symbol,wfdb_description=definitions.get(symbol,'UNRESOLVED'),
            project_group=group,class_index=CLASS_INDICES[group] if group else '',policy=policy,
            train_raw_annotations=raw_counts['train'][symbol],validation_raw_annotations=raw_counts['validation'][symbol],
            train_exported_beats=export_counts['train'][symbol],validation_exported_beats=export_counts['validation'][symbol],
            ambiguity_or_protocol_note=note,definition_source='installed WFDB ann_label_table; official WFDB annotation-code guide',
            grouping_source='Existing DEFAULT_MAPPING; Das & Ari 2014 Table 1, not EC57 compliance certification'))
    return result


def select_examples(rows,category,limit=3):
    """Near-median predicted score per record/symbol; deterministic diversity first."""
    def matches(r):
        t,p=r['true_class'],r['predicted_class']
        return {'S_to_N':t=='S' and p=='N','Q_to_N':t=='Q' and p=='N',
                'F_false_positive':t!='F' and p=='F','V_correct':t==p=='V',
                'N_correct':t==p=='N','F_correct':t==p=='F'}[category]
    selected=[r for r in rows if matches(r)]
    if not selected:return []
    median=float(np.median([r['score'] for r in selected]))
    order=sorted(selected,key=lambda r:(abs(r['score']-median),r['record_id'],r['annotation_position']))
    # Cover important source strata rather than highest confidence/cherry-picked signals.
    priorities={'S_to_N':[('symbol','A'),('symbol','a')],
        'Q_to_N':[('symbol','/'),('symbol','f'),('symbol','Q')],
        'F_false_positive':[('record','201'),('record','202'),('record','210')]}.get(category,[])
    result=[]
    for kind,key in priorities:
        found=next((r for r in order if r not in result and r['original_symbol' if kind=='symbol' else 'record_id']==key),None)
        if found is not None and len(result)<limit:result.append(found)
    for r in order:
        if len(result)>=limit:break
        if r not in result and all(r['record_id']!=a['record_id'] for a in result):result.append(r)
    return result[:limit]


def draw_examples(path,items,arrays_by_record,predictions_by_model,caption):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if not items:raise ValueError('No actual examples to plot')
    fig,axes=plt.subplots(2,len(items),figsize=(5.5*len(items),7.2),sharex=True,squeeze=False,layout='constrained')
    t=(np.arange(252)-90)/360
    for col,row in enumerate(items):
        a=arrays_by_record[row['record_id']]
        loc=np.flatnonzero(a['annotation_positions']==row['annotation_position'])
        if len(loc)!=1:raise ValueError('Nonunique annotation position')
        i=int(loc[0])
        if str(a['original_symbols'][i])!=row['original_symbol'] or CLASS_NAMES[int(a['y'][i])]!=row['true_class']:
            raise ValueError('Plot class does not match exported annotations')
        for ch in range(2):
            ax=axes[ch,col];ax.plot(t,a['X'][i,ch],lw=1.2,color=('#2166ac','#b35806')[ch])
            ax.axvline(0,color='#9e0142',lw=1,ls='--',label='Expert annotation (index 90)')
            ax.scatter([0],[a['X'][i,ch,90]],s=13,color='#9e0142',zorder=3)
            ax.set_ylabel(f'{a["channel_names"][ch]} — Z-score (dimensionless)');ax.grid(alpha=.25)
            ax.set_xlim(t[0],t[-1]);ax.set_xlabel('Time relative to expert annotation (s)')
            if ch==0:
                scores=[]
                for model in ('A','B'):
                    p=predictions_by_model[model][row['source_index']]
                    scores.append(f'{model}: {p["predicted_class"]}, uncalibrated Softmax score={p["score"]:.3f}')
                ax.set_title(f'{row["partition"].upper()} | rec {row["record_id"]} | {row["patient_group"]}\n'
                    f'original symbol {row["original_symbol"]!r} | true group {row["true_class"]} | sample {row["annotation_position"]}\n'
                    +'\n'.join(scores),fontsize=9)
    fig.suptitle(caption+'\nAlready filtered, per-record normalized beat windows; not raw continuous ECG',fontsize=12)
    with Path(path).open('xb') as f:fig.savefig(f,dpi=300,format='png')
    plt.close(fig)
