"""MIT-BIH export inventory and a frozen patient-group protocol, separate from PTB-XL.

Split design uses annotation COUNT metadata only, before model fitting. No ECG
arrays or model outputs are needed to propose/validate a split.
"""
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from numbers import Integral
from pathlib import Path
import numpy as np
from .prepare_dataset import (load_patient_groups, load_record_shard, sha256_file,
    FORMAT_VERSION, MAPPING_VERSION, PHASE9_COUNTS)
from .label_mapping import CLASS_NAMES

DEFAULT_ROOT=Path('data/processed/ecg_dataset_v1')
DEFAULT_SPLIT=Path('configs/mitbih_patient_split_v1.json')
PARTITIONS=('train','validation','test')


@dataclass(frozen=True)
class BeatShard:
    record_id: str
    patient_group: str
    file: str
    count: int
    class_counts: tuple[int,...]
    channel_names: tuple[str,...]
    sha256: str
    global_start: int


class MITBIHExportIndex:
    """Metadata-only index; NPZ arrays are loaded explicitly and one shard at a time."""
    def __init__(self,root=DEFAULT_ROOT,*,require_full_cohort=True):
        self.root=Path(root).resolve();self.manifest_path=self.root/'dataset_metadata.json'
        self.manifest=json.loads(self.manifest_path.read_text())
        m=self.manifest
        if m.get('status')!='complete' or m.get('format_version')!=FORMAT_VERSION:
            raise ValueError('A completed MIT-BIH sharded export is required')
        if (m.get('class_names')!=list(CLASS_NAMES) or m.get('sequence_length')!=252
                or m.get('channels')!=2 or m.get('sampling_frequency')!=360
                or m.get('mapping_version')!=MAPPING_VERSION):
            raise ValueError('MIT-BIH tensor/label/preprocessing contract differs')
        self.manifest_sha256=sha256_file(self.manifest_path)
        for name,digest in m.get('artifact_sha256',{}).items():
            path=(self.root/name).resolve()
            if not path.is_relative_to(self.root) or sha256_file(path)!=digest:
                raise ValueError('Export metadata checksum mismatch')
        patient_path=(self.root/m['patient_mapping_file']).resolve()
        if not patient_path.is_relative_to(self.root):raise ValueError('Patient metadata escapes export')
        groups=load_patient_groups(patient_path) if require_full_cohort else json.loads(patient_path.read_text())
        self.patient_metadata=groups
        entries=[];start=0
        for row in m['shards']:
            rid=row['record_id'];group=row['patient_group'];count=row['count']
            name=row['file'];digest=row['sha256']
            if (not isinstance(rid,str) or not rid.isdigit() or len(rid)!=3
                    or name!=f'recordings/{rid}.npz' or isinstance(count,bool)
                    or not isinstance(count,int) or count<1):raise ValueError('Invalid shard identity/count')
            path=(self.root/name).resolve()
            if not path.is_relative_to(self.root) or not path.is_file():raise FileNotFoundError(f'Missing shard {rid}: {path}')
            if (not isinstance(digest,str) or len(digest)!=64
                    or any(c not in '0123456789abcdef' for c in digest)):
                raise ValueError('Invalid shard SHA256')
            if group!=groups['recordings'][rid]['patient_group'] or not group:
                raise ValueError('Patient group disagrees with verified mapping')
            counts=tuple(row['class_counts'][c] for c in CLASS_NAMES)
            if any(isinstance(v,bool) or not isinstance(v,int) or v<0 for v in counts) or sum(counts)!=count:
                raise ValueError('Class counts disagree with shard size')
            if row['shape']!=[count,2,252] or row['sampling_frequency']!=360 or len(row['channel_names'])!=2:
                raise ValueError('Invalid shard shape/channel metadata')
            entries.append(BeatShard(rid,group,name,count,counts,tuple(row['channel_names']),digest,start));start+=count
        if (len({e.record_id for e in entries})!=len(entries)
                or [e.record_id for e in entries]!=sorted(e.record_id for e in entries)):
            raise ValueError('Record IDs must be unique and sorted')
        self.entries=tuple(entries);self.by_record={e.record_id:e for e in entries}
        self.groups=tuple(sorted({e.patient_group for e in entries}))
        totals=np.sum([e.class_counts for e in entries],axis=0).astype(np.int64)
        if (len(entries)!=m['record_count'] or len(self.groups)!=m['patient_group_count']
                or start!=m['segment_count'] or totals.tolist()!=[m['class_counts'][c] for c in CLASS_NAMES]):
            raise ValueError('Manifest total counts disagree')
        self.segment_count=start
        if require_full_cohort and (len(entries)!=48 or len(self.groups)!=47 or start!=109452
                or totals.tolist()!=[PHASE9_COUNTS[c] for c in CLASS_NAMES]):
            raise ValueError('Expected completed 48-record, 47-group, 109452-beat cohort')
        if '201' in self.by_record and '202' in self.by_record:
            if self.by_record['201'].patient_group!=self.by_record['202'].patient_group:
                raise ValueError('Records 201 and 202 must share one subject group')
        self.read_records=[]

    def group_counts(self):
        result={g:np.zeros(5,dtype=np.int64) for g in self.groups}
        for e in self.entries:result[e.patient_group]+=e.class_counts
        return result

    def load_shard(self,record_id,split,partition,*,allow_final_test=False):
        """Guard partition BEFORE hashing/reading a waveform-bearing NPZ."""
        if partition not in PARTITIONS:raise ValueError('Unknown partition')
        if not isinstance(allow_final_test,bool):raise ValueError('Final-test opt-in must be an explicit boolean')
        if partition=='test' and not allow_final_test:raise PermissionError('Final MIT-BIH test is locked')
        validate_split(self,split)
        entry=self.by_record[record_id]
        if entry.patient_group not in split['groups'][partition]:
            raise PermissionError('Source group does not belong to this partition')
        path=self.root/entry.file
        if sha256_file(path)!=entry.sha256:raise ValueError(f'MIT-BIH shard SHA256 differs: {record_id}')
        arrays=load_record_shard(path)
        if (len(arrays['y'])!=entry.count or not np.all(arrays['record_ids']==record_id)
                or not np.all(arrays['patient_groups']==entry.patient_group)
                or arrays['channel_names'].tolist()!=list(entry.channel_names)
                or np.bincount(arrays['y'],minlength=5).tolist()!=list(entry.class_counts)):
            raise ValueError(f'Shard arrays/metadata disagree: {record_id}')
        self.read_records.append(record_id)
        return arrays


def inventory(index):
    counts=index.group_counts();totals=np.sum(list(counts.values()),axis=0)
    per=[]
    for group,values in counts.items():
        records=[e.record_id for e in index.entries if e.patient_group==group]
        per.append({'patient_group':group,'record_ids':records,'beats':int(values.sum()),
                    'counts':dict(zip(CLASS_NAMES,map(int,values)))})
    classes=[]
    for i,c in enumerate(CLASS_NAMES):
        ranked=sorted(((g,int(v[i])) for g,v in counts.items() if v[i]),key=lambda r:(-r[1],r[0]))
        classes.append({'class':c,'beats':int(totals[i]),'supporting_groups':len(ranked),
            'groups_descending':ranked,'largest_group_share':ranked[0][1]/int(totals[i])})
    return {'per_group':per,'per_class':classes,'features_read':False}


def validate_split(index,split):
    """Reject missing/duplicated groups, stale source identities and altered counts."""
    if split.get('manifest_sha256')!=index.manifest_sha256 or split.get('class_names')!=list(CLASS_NAMES):
        raise ValueError('Split belongs to another export/label contract')
    if set(split['groups'])!=set(PARTITIONS):raise ValueError('All three partitions required')
    values=[split['groups'][p] for p in PARTITIONS]
    if any(not isinstance(v,list) or not v or len(v)!=len(set(v)) for v in values):
        raise ValueError('Nonempty unique group lists required')
    flattened=[g for group in values for g in group]
    if len(flattened)!=len(set(flattened)) or set(flattened)!=set(index.groups):
        raise ValueError('Patient overlap or incomplete group coverage')
    group_counts=index.group_counts();result={}
    for p in PARTITIONS:
        groups=set(split['groups'][p]);counts=np.sum([group_counts[g] for g in groups],axis=0)
        records=[e.record_id for e in index.entries if e.patient_group in groups]
        result[p]={'group_count':len(groups),'record_ids':records,'record_count':len(records),
                   'beats':int(counts.sum()),'class_counts':dict(zip(CLASS_NAMES,map(int,counts))),
                   'supporting_groups':{c:int(sum(group_counts[g][i]>0 for g in groups)) for i,c in enumerate(CLASS_NAMES)}}
    if split.get('summary') is not None and split['summary']!=result:raise ValueError('Stored split counts differ')
    return result


def propose_split(index,*,seed=42,candidates=20000):
    """Metadata-only, single prespecified group search; no model/feature selection.

    29/9/9 subjects (nominal 60/20/20). Largest F subject trains, second tests;
    two remaining F-rich subjects validate, another trains. Four dominant Q
    subjects distribute 2/1/1, with both major Q lead configurations in training.
    This deliberately sacrifices beat-stratification for independent patients.
    """
    if len(index.groups)!=47:raise ValueError('This proposed v1 protocol requires the actual 47-group cohort')
    if isinstance(seed,bool) or not isinstance(seed,int) or not 0<=seed<2**32:raise ValueError('Invalid seed')
    if isinstance(candidates,bool) or not isinstance(candidates,int) or candidates<1:raise ValueError('Invalid candidate count')
    groups=list(index.groups);counts=index.group_counts()
    f=sorted((g for g in groups if counts[g][3]),key=lambda g:(-counts[g][3],g))
    q=sorted((g for g in groups if counts[g][4]),key=lambda g:(-counts[g][4],g))[:4]
    anchors={'train':{f[0],f[4]},'validation':{f[2],f[3]},'test':{f[1]}}
    signatures={g:{e.channel_names for e in index.entries if e.patient_group==g} for g in q}
    q_first=q[0];q_other=next(g for g in q[1:] if signatures[g]!=signatures[q_first])
    anchors['train'].update((q_first,q_other))
    remaining_q=sorted(set(q)-{q_first,q_other})
    anchors['validation'].add(remaining_q[0]);anchors['test'].add(remaining_q[1])
    anchors['train'].add(max(groups,key=lambda g:(counts[g][1],g)))
    assigned=[g for p in PARTITIONS for g in anchors[p]]
    if len(assigned)!=len(set(assigned)):raise ValueError('Rare-class anchors conflict; review cohort')
    remaining=sorted(set(groups)-set(assigned));sizes={'train':29,'validation':9,'test':9}
    slots=[sizes[p]-len(anchors[p]) for p in PARTITIONS]
    data=np.array([counts[g] for g in groups]);totals=data.sum(axis=0)
    group_pos={g:i for i,g in enumerate(groups)};fractions=np.array([.6,.2,.2])
    rng=np.random.default_rng(seed);best=None;feasible=0
    for _ in range(candidates):
        shuffled=rng.permutation(remaining).tolist();assignment={};offset=0
        for p,n in zip(PARTITIONS,slots):
            assignment[p]=sorted(anchors[p]|set(shuffled[offset:offset+n]));offset+=n
        part=np.array([data[[group_pos[g] for g in assignment[p]]].sum(axis=0) for p in PARTITIONS])
        support=np.array([(data[[group_pos[g] for g in assignment[p]]]>0).sum(axis=0) for p in PARTITIONS])
        if np.any(part==0) or np.any(support<np.array([3,3,3,2,2])):continue
        feasible+=1
        objective=float(np.square(part/totals-fractions[:,None]).sum()
                        +.5*np.square(part.sum(axis=1)/totals.sum()-fractions).sum())
        key=(objective,tuple(tuple(assignment[p]) for p in PARTITIONS))
        if best is None or key<best[0]:best=(key,assignment)
    if best is None:raise ValueError('No feasible patient split; do not relax constraints silently')
    split={'schema':'mitbih-patient-split-v1','manifest_sha256':index.manifest_sha256,
        'class_names':list(CLASS_NAMES),'seed':seed,'candidate_count':candidates,
        'nominal_beat_fractions':[.6,.2,.2],'target_group_counts':sizes,
        'method':'Count-metadata-only constrained group allocation; no waveform/model optimization',
        'rare_class_anchors':{p:sorted(v) for p,v in anchors.items()},
        'minimum_class_supporting_groups':[3,3,3,2,2],
        'feasible_candidates':feasible,'objective':best[0][0],'groups':best[1],
        'patient_identity_scope':'technical dataset groups, not real patient identities',
        'test_status':'locked; no model evaluation in Phase 20'}
    split['summary']=validate_split(index,split)
    return split


def load_split(index,path=DEFAULT_SPLIT):
    split=json.loads(Path(path).read_text());validate_split(index,split);return split


def training_class_weights(index,split,training_groups):
    """Balanced CE n_train/(5*n_class); require explicit EXACT training groups."""
    validate_split(index,split)
    if not isinstance(training_groups,(list,tuple)) or not training_groups:
        raise ValueError('Explicit training-group list required')
    if len(training_groups)!=len(set(training_groups)) or set(training_groups)!=set(split['groups']['train']):
        raise ValueError('Only the explicit complete training patient groups are allowed')
    counts=index.group_counts();total=np.sum([counts[g] for g in training_groups],axis=0)
    if np.any(total==0):raise ValueError('Every class must be represented in training')
    return (total.sum()/(5*total.astype(np.float64))).astype(np.float32)
