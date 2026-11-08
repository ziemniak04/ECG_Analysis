"""Reproducible, per-record NPZ export. No training, balancing or dataset splitting."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
from collections import Counter
import numpy as np
from .data_loader import DEFAULT_DATA_DIR, list_recordings, locate_data_dir, load_recording
from .segmentation import segment_recording
from .label_mapping import CLASS_NAMES, DEFAULT_MAPPING, map_segments
from .class_imbalance import analyze_labeled, describe_counts

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATIENT_FILE = PROJECT_ROOT / 'metadata/patient_groups.json'
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / 'data/processed/ecg_dataset_v1'
FORMAT_VERSION = 'ecg-sharded-npz-v1'
MAPPING_VERSION = 'mitdb-aami-style-v1'
PHASE9_COUNTS = {'N':90594,'S':2781,'V':7235,'F':802,'Q':8040}
PER_BEAT_FIELDS = ('X','y','record_ids','patient_groups','annotation_positions',
                   'original_symbols','annotation_indices','annotation_aux_notes')


def sha256_file(path):
    """Hash in bounded chunks without loading an entire source file."""
    digest=hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda:source.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes(root):
    """Snapshot every regular file in the local dataset directory (including sidecars)."""
    root=Path(root)
    return {p.relative_to(root).as_posix():sha256_file(p) for p in sorted(root.rglob('*')) if p.is_file()}


def load_patient_groups(path=DEFAULT_PATIENT_FILE):
    """Read the explicit, source-attributed grouping; reject inconsistent relationships."""
    with Path(path).open() as source:
        metadata=json.load(source)
    records=metadata['recordings']
    if len(records)!=48 or metadata['documented_subject_count']!=47:
        raise ValueError('Expected documented 48 recordings / 47 subjects')
    groups={rid:row['patient_group'] for rid,row in records.items()}
    if any(not isinstance(g,str) or not g for g in groups.values()):
        raise ValueError('Unresolved patient group')
    if groups.get('201')!=groups.get('202') or len(set(groups.values()))!=47:
        raise ValueError('Patient grouping disagrees with documented subject relationships')
    pairs={group:sorted(rid for rid,g in groups.items() if g==group) for group in set(groups.values())}
    if sorted(members for members in pairs.values() if len(members)>1)!=[['201','202']]:
        raise ValueError('Unexpected shared-subject relationship')
    if any(row.get('grouping_verified_within_dataset') is not True for row in records.values()):
        raise ValueError('Patient grouping must have a verified dataset basis')
    return metadata


def write_json(path, data):
    with Path(path).open('x',encoding='utf-8') as output:
        json.dump(data,output,indent=2,sort_keys=True,allow_nan=False)


def write_csv(path, rows):
    with Path(path).open('x',newline='',encoding='utf-8') as output:
        writer=csv.DictWriter(output,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_shard(arrays):
    """Validate shape/dtype/alignment, classes, symbols and offline preprocessing metadata."""
    required=set(PER_BEAT_FIELDS)|{'class_names','sampling_frequency','channel_names','preprocessing_config'}
    if not required.issubset(arrays):
        raise ValueError('Missing dataset fields')
    X=arrays['X']
    if X.ndim!=3 or X.shape[1:]!=(2,252) or X.dtype!=np.float32 or not np.isfinite(X).all():
        raise ValueError('Expected finite float32 X [beats, 2, 252]')
    n=len(X)
    if any(len(arrays[name])!=n for name in PER_BEAT_FIELDS):
        raise ValueError('Per-beat array lengths disagree')
    if any(arrays[name].ndim!=1 for name in PER_BEAT_FIELDS if name!='X'):
        raise ValueError('Per-beat metadata must be one-dimensional')
    y=arrays['y']
    if y.dtype!=np.int64 or np.any(y<0) or np.any(y>=5):
        raise ValueError('Expected int64 class labels 0..4')
    if arrays['annotation_positions'].dtype!=np.int64 or arrays['annotation_indices'].dtype!=np.int64:
        raise ValueError('Annotation positions/indices must be int64')
    for name in ('record_ids','patient_groups','original_symbols','annotation_aux_notes','channel_names'):
        if arrays[name].dtype.kind!='U':
            raise ValueError('Text metadata must be Unicode, without pickle')
    if arrays['class_names'].tolist()!=list(CLASS_NAMES) or arrays['channel_names'].shape!=(2,):
        raise ValueError('Invalid class or channel metadata')
    if float(arrays['sampling_frequency'])!=360:
        raise ValueError('Export contract requires original MIT-BIH sampling rate 360 Hz')
    expected=[DEFAULT_MAPPING.get(s) for s in arrays['original_symbols']]
    if any(s is None for s in expected) or expected!=[CLASS_NAMES[i] for i in y]:
        raise ValueError('Original symbols and class labels disagree')
    if n and (np.any(arrays['patient_groups']=='') or len(set(arrays['record_ids'].tolist()))!=1):
        raise ValueError('Expected one source recording and resolved groups per shard')
    config=json.loads(str(arrays['preprocessing_config']))
    if config['mapping_version']!=MAPPING_VERSION or config['normalization']['scope']!='recording':
        raise ValueError('Unexpected mapping/normalization contract')
    if config['normalization']['method']!='zscore' or config['sequence_length']!=252:
        raise ValueError('Unexpected normalization/segmentation configuration')
    return n


def load_record_shard(path):
    """Load one recording safely, retaining native float32/int64 arrays (no PyTorch dependency)."""
    with np.load(path,allow_pickle=False) as archive:
        arrays={key:archive[key] for key in archive.files}
    validate_shard(arrays)
    return arrays


def export_dataset(root=DEFAULT_DATA_DIR, output_dir=DEFAULT_OUTPUT_DIR, *,
                   patient_file=DEFAULT_PATIENT_FILE, record_ids=None, progress=None):
    """Export all records by default, at most one full recording resident at a time.

    A new directory is mandatory. dataset_metadata.json is written LAST and marks
    successful completion. A failed partial directory is left for review, not reused.
    Explicit record_ids are supported for small verification exports, clearly scoped.
    """
    data_dir=locate_data_dir(root)
    patient_metadata=load_patient_groups(patient_file)
    available=list_recordings(data_dir)
    known=patient_metadata['recordings']
    ids=available if record_ids is None else sorted(record_ids)
    if not ids or len(set(ids))!=len(ids) or any(rid not in available or rid not in known for rid in ids):
        raise ValueError('Need unique, available recordings with documented patient groups')
    if record_ids is None and set(available)!=set(known):
        raise ValueError('Complete export requires all 48 documented recordings')
    output_dir=Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=False)
    (output_dir/'recordings').mkdir()
    before=source_hashes(data_dir)
    write_json(output_dir/'patient_groups.json',patient_metadata)
    shards=[]
    leads=[]
    counts=Counter()
    exclusions={'total_annotations':0,'excluded_nonheartbeat':0,'skipped_boundary':0}
    for rid in ids:
        record=load_recording(rid,data_dir)
        if record.sampling_frequency!=360 or record.signals.shape[1]!=2:
            raise ValueError(f'{rid}: unexpected fs/channel count; no resampling/relabeling is allowed')
        group=known[rid]['patient_group']
        labeled=map_segments(segment_recording(record,patient_group_id=group))
        audit=analyze_labeled(labeled)
        segments=labeled.segments
        config={**segments.configuration,'mapping_version':MAPPING_VERSION,
                'class_mapping':dict(DEFAULT_MAPPING),'class_names':list(CLASS_NAMES),
                'output_dtype':'float32','patient_group_mapping_version':patient_metadata['version']}
        # Convert only one recording's ECG windows; all other arrays stay small.
        arrays={'X':labeled.X.astype(np.float32),'y':labeled.y.astype(np.int64,copy=False),
            'record_ids':segments.record_ids,'patient_groups':segments.patient_group_ids,
            'annotation_positions':segments.annotation_positions,'original_symbols':segments.annotation_symbols,
            'annotation_indices':segments.annotation_indices,'annotation_aux_notes':segments.annotation_aux_notes,
            'class_names':np.array(CLASS_NAMES),'sampling_frequency':np.array(record.sampling_frequency),
            'channel_names':np.array(record.channel_names),
            'preprocessing_config':np.array(json.dumps(config,sort_keys=True,allow_nan=False))}
        validate_shard(arrays)
        relative=Path('recordings')/f'{rid}.npz'
        path=output_dir/relative
        with path.open('xb') as target:
            np.savez_compressed(target,**arrays)
        restored=load_record_shard(path)
        for key in arrays:
            if not np.array_equal(arrays[key],restored[key]):
                raise ValueError(f'{rid}: round-trip mismatch for {key}')
        counts.update(audit['class_counts'])
        for key in exclusions:
            exclusions[key]+=segments.report[key]
        shards.append({'record_id':rid,'file':relative.as_posix(),'count':len(labeled.y),
            'patient_group':group,'channel_names':list(record.channel_names),'source_units':list(record.units),
            'sampling_frequency':record.sampling_frequency,'shape':list(arrays['X'].shape),
            'class_counts':audit['class_counts'],'segmentation_report':segments.report,
            'sha256':sha256_file(path)})
        leads.append({'record_id':rid,'channel_0':record.channel_names[0],'channel_1':record.channel_names[1],
                      'source_unit_0':record.units[0],'source_unit_1':record.units[1],
                      'sampling_frequency':record.sampling_frequency})
        if progress:
            progress(rid,len(labeled.y))
        del arrays,restored,labeled,segments,record
    counts={name:int(counts[name]) for name in CLASS_NAMES}
    full=set(ids)==set(known)
    if full and counts!=PHASE9_COUNTS:
        raise ValueError(f'Phase 9 discrepancy: expected {PHASE9_COUNTS}, obtained {counts}')
    after=source_hashes(data_dir)
    if before!=after:
        raise ValueError('Original database changed during preparation')
    distribution=describe_counts(counts)
    write_csv(output_dir/'class_distribution.csv',[
        {'class_name':name,'class_index':i,'count':counts[name],'percentage':distribution['class_percentages'][name]}
        for i,name in enumerate(CLASS_NAMES)])
    write_csv(output_dir/'lead_configuration.csv',leads)
    versions={name:importlib.metadata.version(name) for name in ('numpy','scipy','wfdb','pandas','matplotlib')}
    sources=['data_loader.py','preprocessing.py','segmentation.py','label_mapping.py','class_imbalance.py','prepare_dataset.py','visualization.py']
    manifest={'format_version':FORMAT_VERSION,'status':'complete','scope':'all_48_records' if full else 'explicit_subset',
        'record_ids':ids,'record_count':len(ids),'patient_group_count':len({known[rid]['patient_group'] for rid in ids}),
        'segment_count':sum(counts.values()),'channels':2,'sequence_length':252,'sampling_frequency':360.0,
        'X_dtype':'float32','y_dtype':'int64','X_units':'dimensionless','class_names':list(CLASS_NAMES),
        'class_indices':{name:i for i,name in enumerate(CLASS_NAMES)},'mapping_version':MAPPING_VERSION,
        'mapping':dict(DEFAULT_MAPPING),'class_counts':counts,'distribution':distribution,
        'phase9_counts_match':counts==PHASE9_COUNTS if full else None,'annotation_audit':exclusions,
        'shards':shards,'normalization_scope':'recording_per_channel_after_filtering_before_segmentation',
        'preprocessing_config_location':'Each NPZ preprocessing_config scalar JSON includes fitted statistics and all parameters',
        'channel_policy':'Original header order and lead names preserved; channel indices are not anatomically equivalent across records',
        'patient_mapping_file':'patient_groups.json','patient_mapping_version':patient_metadata['version'],
        'source_files_sha256':before,'source_files_unchanged':True,
        'code_sha256':{name:sha256_file(Path(__file__).parent/name) for name in sources},
        'python_version':platform.python_version(),'package_versions':versions,
        'artifact_sha256':{name:sha256_file(output_dir/name) for name in
                            ('patient_groups.json','class_distribution.csv','lead_configuration.csv')},
        'ordering':'record_id ascending; original annotation order within each record',
        'no_balancing':True,'no_split':True,'patient_identity_scope':'verified dataset grouping, not real identities'}
    write_json(output_dir/'dataset_metadata.json',manifest)
    return manifest


def verify_export(output_dir=DEFAULT_OUTPUT_DIR, *, source_root=None):
    """Read and audit every exported shard, checksums, source annotations and lead names."""
    import wfdb
    output_dir=Path(output_dir)
    with (output_dir/'dataset_metadata.json').open() as source:
        metadata=json.load(source)
    if metadata['status']!='complete' or metadata['format_version']!=FORMAT_VERSION:
        raise ValueError('Incomplete or unsupported dataset')
    groups=load_patient_groups(output_dir/metadata['patient_mapping_file'])['recordings']
    counts=Counter()
    total=0
    data_dir=locate_data_dir(source_root) if source_root is not None else None
    for name,digest in metadata['artifact_sha256'].items():
        if sha256_file(output_dir/name)!=digest:
            raise ValueError(f'Artifact checksum mismatch: {name}')
    for shard in metadata['shards']:
        path=output_dir/shard['file']
        if sha256_file(path)!=shard['sha256']:
            raise ValueError('Shard checksum mismatch')
        arrays=load_record_shard(path)
        n=len(arrays['y'])
        rid=shard['record_id']
        if n!=shard['count'] or not np.all(arrays['record_ids']==rid):
            raise ValueError('Record/row count mismatch')
        if not np.all(arrays['patient_groups']==groups[rid]['patient_group']):
            raise ValueError('Patient group mismatch')
        if arrays['channel_names'].tolist()!=shard['channel_names']:
            raise ValueError('Channel metadata mismatch')
        if data_dir is not None:
            annotation=wfdb.rdann(str(data_dir/rid),'atr')
            indices=arrays['annotation_indices']
            if not np.array_equal(arrays['annotation_positions'],np.asarray(annotation.sample)[indices]):
                raise ValueError('Original annotation positions changed')
            if arrays['original_symbols'].tolist()!=[annotation.symbol[i] for i in indices]:
                raise ValueError('Original annotation symbols changed')
            header=wfdb.rdheader(str(data_dir/rid))
            if arrays['channel_names'].tolist()!=header.sig_name or float(arrays['sampling_frequency'])!=header.fs:
                raise ValueError('Original channel/fs metadata changed')
        counts.update({name:int(np.count_nonzero(arrays['y']==i)) for i,name in enumerate(CLASS_NAMES)})
        total+=n
    if dict(counts)!=metadata['class_counts'] or total!=metadata['segment_count']:
        raise ValueError('Final exported counts disagree with manifest')
    if len(metadata['shards'])!=metadata['record_count'] or len(set(metadata['record_ids']))!=metadata['record_count']:
        raise ValueError('Record count mismatch')
    if len({groups[rid]['patient_group'] for rid in metadata['record_ids']})!=metadata['patient_group_count']:
        raise ValueError('Subject group count mismatch')
    if metadata['scope']=='all_48_records' and (dict(counts)!=PHASE9_COUNTS or metadata['record_count']!=48):
        raise ValueError('Full dataset differs from Phase 9')
    if data_dir is not None and source_hashes(data_dir)!=metadata['source_files_sha256']:
        raise ValueError('Original source-file checksums changed')
    return {'segment_count':total,'record_count':metadata['record_count'],
            'patient_group_count':metadata['patient_group_count'],'class_counts':dict(counts),
            'source_files_verified':data_dir is not None}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=DEFAULT_DATA_DIR)
    parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    if not args.verify_only:
        export_dataset(args.data_dir,args.output_dir,progress=lambda rid,n:print(f'{rid}: exported {n} segments',flush=True))
    print(json.dumps(verify_export(args.output_dir,source_root=args.data_dir),indent=2))


if __name__=='__main__':
    main()
