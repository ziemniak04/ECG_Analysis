"""Versioned RR-sidecar export and lazy ECG/RR join; no waveform duplication."""
from collections import OrderedDict
import json
from pathlib import Path
import numpy as np
import torch
import wfdb
from torch.utils.data import Dataset
from .mitbih_protocol import MITBIHExportIndex,load_split,validate_split,DEFAULT_SPLIT
from .mitbih_torch import MITBIHBeatDataset,collate_mitbih
from .mitbih_readiness import verify_frozen_protocol
from .mitbih_rr import rr_features,beat_ids,fit_training_transform,transform_rr,validate_raw
from .mitbih_rr_config import RRFeatureConfig,RR_FEATURE_NAMES,RAW_RR_NAMES
from .visualization import BEAT_SYMBOLS
from .data_loader import DEFAULT_DATA_DIR
from .prepare_dataset import sha256_file
from .ptbxl_training import write_json_new

DEFAULT_RR_ROOT=Path('data/processed/mitbih_rr_context_v1')


def partition_for(index,split,rid):
    for p in ('train','validation'):
        if index.by_record[rid].patient_group in split['groups'][p]:return p
    raise PermissionError('Final-test timing and waveforms are deferred')


def export_rr(output=DEFAULT_RR_ROOT):
    """Build only TRAIN/VAL sidecars. Reads metadata fields of X shards, never X."""
    output=Path(output)
    if output.exists():raise FileExistsError('Use a new versioned RR output directory')
    index=MITBIHExportIndex();split=load_split(index);verify_frozen_protocol(index,split)
    config=RRFeatureConfig();pending={};source_checks={};training=[];training_masks=[];training_groups=[]
    for e in index.entries:
        if e.patient_group in split['groups']['test']:continue
        partition=partition_for(index,split,e.record_id)
        base=DEFAULT_DATA_DIR/e.record_id
        for ext in ('hea','atr'):
            path=base.with_suffix('.'+ext);digest=sha256_file(path)
            if digest!=index.manifest['source_files_sha256'][path.name]:raise ValueError('Raw annotation/header changed')
            source_checks[str(path)]=digest
        header=wfdb.rdheader(str(base));annotation=wfdb.rdann(str(base),'atr')
        if header.fs!=index.manifest['sampling_frequency']:raise ValueError('Sampling-rate mismatch')
        # Decode metadata only: original ECG X is not opened as an array here.
        source=index.root/e.file
        if sha256_file(source)!=e.sha256:raise ValueError('Original shard checksum differs')
        with np.load(source,allow_pickle=False) as z:
            arrays={k:z[k] for k in ('y','record_ids','patient_groups','annotation_positions','annotation_indices','original_symbols')}
        ai=arrays['annotation_indices'];positions=np.asarray(annotation.sample,dtype=np.int64)
        symbols=np.array(annotation.symbol)
        if (len(ai)!=e.count or not np.array_equal(positions[ai],arrays['annotation_positions'])
                or not np.array_equal(symbols[ai],arrays['original_symbols'])
                or not np.all(arrays['record_ids']==e.record_id) or not np.all(arrays['patient_groups']==e.patient_group)):
            raise ValueError('Source annotation/identity alignment differs')
        heartbeat_positions=positions[np.array([s in BEAT_SYMBOLS for s in annotation.symbol])]
        raw,valid=rr_features(heartbeat_positions,arrays['annotation_positions'],header.fs,config)
        arrays.update(rr_raw=raw,rr_valid=valid,beat_ids=beat_ids(e.record_id,ai,arrays['annotation_positions']),
            source_rows=np.arange(e.count,dtype=np.int64),global_source_indices=e.global_start+np.arange(e.count,dtype=np.int64),
            sampling_frequency=np.array(header.fs,dtype=np.float64),feature_names=np.array(RR_FEATURE_NAMES))
        pending[e.record_id]=(e,partition,arrays)
        if partition=='train':
            training.append(raw);training_masks.append(valid);training_groups.extend([e.patient_group]*e.count)
    state=fit_training_transform(np.concatenate(training),np.concatenate(training_masks),partition='train',
        patient_groups=training_groups,expected_training_groups=split['groups']['train'],manifest_sha256=index.manifest_sha256)
    output.mkdir(parents=True,exist_ok=False);(output/'recordings').mkdir()
    write_json_new(output/'rr_configuration.json',dict(**config.to_dict(),raw_feature_names=list(RAW_RR_NAMES),
        model_feature_names=list(RR_FEATURE_NAMES),context_policy='past_only; no next RR',
        valid_heartbeat_symbols=sorted(BEAT_SYMBOLS),baseline_excludes_current_previous_interval=True,
        no_class_labels_in_features=True,source_context='all valid ATR beats within SAME record, including boundary-skipped windows',
        missing_raw_value=0.,missing_model_value='TRAIN median, then TRAIN valid-only zscore; explicit binary masks',
        recording_boundaries='never join, including 201/202',no_physiological_threshold_clipping=True))
    write_json_new(output/'training_transform.json',state)
    manifest_rows=[];summary={}
    for rid,(e,partition,arrays) in pending.items():
        arrays['RR']=transform_rr(arrays['rr_raw'],arrays['rr_valid'],state)
        path=output/'recordings'/f'{rid}_rr.npz'
        with path.open('xb') as f:np.savez_compressed(f,**arrays)
        manifest_rows.append(dict(record_id=rid,patient_group=e.patient_group,partition=partition,count=e.count,
            file=str(path.relative_to(output)),sha256=sha256_file(path),source_X_file=str((index.root/e.file).relative_to(Path.cwd())),
            source_X_sha256=e.sha256,source_annotation_sha256=source_checks[str((DEFAULT_DATA_DIR/rid).with_suffix('.atr'))],
            sampling_frequency=float(arrays['sampling_frequency']),shape=list(arrays['RR'].shape)))
        row=summary.setdefault(partition,dict(beats=0,missing_counts=[0,0,0],class_counts=[0]*5))
        row['beats']+=e.count
        row['missing_counts']=(np.array(row['missing_counts'])+(~arrays['rr_valid']).sum(0)).tolist()
        row['class_counts']=(np.array(row['class_counts'])+np.bincount(arrays['y'],minlength=5)).tolist()
    for partition in ('train','validation'):
        if summary[partition]['beats']!=split['summary'][partition]['beats']:
            raise ValueError('Full partition count mismatch')
    metadata=dict(schema='mitbih-rr-sidecar-v1',status='complete',records=manifest_rows,summary=summary,
        total_beats=sum(r['count'] for r in manifest_rows),record_count=len(manifest_rows),
        class_names=list('NSVFQ'),RR_dtype='float32',RR_width=6,source_X_reused=True,
        source_manifest_sha256=index.manifest_sha256,frozen_split_sha256=sha256_file(DEFAULT_SPLIT),
        artifact_sha256={p.name:sha256_file(p) for p in (output/'rr_configuration.json',output/'training_transform.json')},
        raw_source_sha256=source_checks,feature_configuration=config.to_dict(),final_test_status='Not read; no RR artifacts until separate approval')
    write_json_new(output/'manifest.json',metadata)  # Complete marker written LAST.
    return metadata


class MITBIHRRDataset(Dataset):
    """Lazy aligned view of existing ECG windows plus record-level RR sidecars."""
    def __init__(self,rr_root=DEFAULT_RR_ROOT,*,partition='train',index=None,split=None,max_cached_shards=2):
        if partition not in ('train','validation'):raise PermissionError('Phase-24 adapter never opens final test')
        self.base=MITBIHBeatDataset(index,split,partition=partition,max_cached_shards=max_cached_shards)
        self.root=Path(rr_root);self.partition=partition;self.cache=OrderedDict();self.max_cached_shards=max_cached_shards
        self.manifest=json.loads((self.root/'manifest.json').read_text())
        m=self.manifest
        if (m.get('status')!='complete' or m.get('schema')!='mitbih-rr-sidecar-v1' or m.get('RR_width')!=6 or m.get('feature_configuration')!=RRFeatureConfig().to_dict()
                or m.get('class_names')!=list('NSVFQ') or m.get('source_manifest_sha256')!=self.base.index.manifest_sha256):
            raise ValueError('Wrong RR/source contract')
        # The frozen split is stored separately; validate exact file identity for real data.
        if len(self.base.index.groups)==47 and m['frozen_split_sha256']!=sha256_file(DEFAULT_SPLIT):
            raise ValueError('RR sidecar belongs to another patient split')
        for name,digest in m['artifact_sha256'].items():
            if name not in ('rr_configuration.json','training_transform.json') or sha256_file(self.root/name)!=digest:
                raise ValueError('RR configuration/transform changed')
        self.transform=json.loads((self.root/'training_transform.json').read_text())
        if set(self.transform['training_groups'])!=set(self.base.split['groups']['train']):raise ValueError('Scaler training groups differ')
        if self.transform['source_manifest_sha256']!=self.base.index.manifest_sha256:raise ValueError('Scaler source differs')
        rows=m['records'];self.rows={r['record_id']:r for r in rows}
        if len(rows)!=len(self.rows):raise ValueError('Duplicate RR record')
        allowed=set(self.base.split['summary']['train']['record_ids']+self.base.split['summary']['validation']['record_ids'])
        if set(self.rows)!=allowed:raise ValueError('Missing/unexpected RR records, including forbidden test IDs')
        for rid,row in self.rows.items():
            e=self.base.index.by_record[rid]
            if (row['source_X_sha256']!=e.sha256 or row['count']!=e.count or row['patient_group']!=e.patient_group
                    or row['partition']!=partition_for(self.base.index,self.base.split,rid)):
                raise ValueError('RR manifest source/partition mismatch')
            path=(self.root/row['file']).resolve()
            if not path.is_relative_to(self.root.resolve()) or not path.is_file():raise FileNotFoundError('Missing/unsafe RR file')

    def __len__(self):return len(self.base)

    def __getitem__(self,position):
        sample=self.base[position];entry,row=self.base.source_location(position)
        path=self.root/self.rows[entry.record_id]['file'];stamp=(path.stat().st_size,path.stat().st_mtime_ns)
        if entry.record_id in self.cache:
            arrays,oldstamp=self.cache.pop(entry.record_id)
            if oldstamp!=stamp:raise ValueError('Cached RR changed')
        else:
            if sha256_file(path)!=self.rows[entry.record_id]['sha256']:raise ValueError('RR checksum mismatch')
            with np.load(path,allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
            original=self.base._arrays(entry)
            for key in ('y','record_ids','patient_groups','annotation_positions','annotation_indices','original_symbols'):
                if not np.array_equal(arrays[key],original[key]):raise ValueError('ECG/RR metadata misalignment')
            if any(arrays[k].dtype!=np.int64 for k in ('y','annotation_positions','annotation_indices','source_rows','global_source_indices')):
                raise ValueError('Integer RR metadata dtype mismatch')
            if any(arrays[k].dtype.kind!='U' for k in ('record_ids','patient_groups','original_symbols','beat_ids','feature_names')):
                raise ValueError('Text RR metadata dtype mismatch')
            if (arrays['RR'].dtype!=np.float32 or arrays['RR'].shape!=(entry.count,6)
                    or arrays['rr_raw'].shape!=(entry.count,3) or arrays['rr_raw'].dtype!=np.float64
                    or arrays['rr_valid'].dtype!=np.bool_ or not np.isfinite(arrays['RR']).all()
                    or arrays['feature_names'].tolist()!=list(RR_FEATURE_NAMES)):
                raise ValueError('RR tensor contract mismatch')
            np.testing.assert_array_equal(arrays['RR'],transform_rr(arrays['rr_raw'],arrays['rr_valid'],self.transform))
            if not np.array_equal(arrays['source_rows'],np.arange(entry.count)) or not np.array_equal(arrays['global_source_indices'],entry.global_start+np.arange(entry.count)):
                raise ValueError('Source order changed')
            if not np.array_equal(arrays['beat_ids'],beat_ids(entry.record_id,arrays['annotation_indices'],arrays['annotation_positions'])):
                raise ValueError('Stable beat identifiers differ')
            if float(arrays['sampling_frequency'])!=self.base.index.manifest['sampling_frequency']:
                raise ValueError('RR sampling frequency differs')
        self.cache[entry.record_id]=(arrays,stamp)
        while len(self.cache)>self.max_cached_shards:self.cache.popitem(last=False)
        return dict(**sample,RR=torch.from_numpy(arrays['RR'][row].copy()),
                    rr_raw=torch.from_numpy(arrays['rr_raw'][row].copy()),rr_valid=torch.from_numpy(arrays['rr_valid'][row].copy()),
                    beat_id=str(arrays['beat_ids'][row]))


def collate_rr(samples):
    batch=collate_mitbih(samples)
    for name in ('RR','rr_raw','rr_valid'):batch[name]=torch.stack([s[name] for s in samples])
    batch['beat_id']=[s['beat_id'] for s in samples]
    return batch
