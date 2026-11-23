"""Read-only PTB-XL export index and official-fold protocol, without PyTorch.

Only manifests/metadata are held in RAM. ECG arrays are loaded one NPZ at a time
through the existing validated reader; no filtering or normalization is repeated.
"""
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from numbers import Integral
from pathlib import Path
import numpy as np
from .ptbxl_metadata import load_ptb_metadata, propose_exclusive_cohort
from .ptbxl_rhythm import CLASS_NAMES, load_pilot_record, record_identity, preprocessing_config

DEFAULT_ROOT = Path('data/processed/ptbxl_rhythm_v1')
DEFAULT_MANIFEST = Path('runs/90b449bf54b649509519747f417b9e5e/PTBXL_DOWNLOAD_MANIFEST.json')
EXPECTED_COUNTS = {'SR':16687, 'AFIB':1484, 'STACH':805, 'SBRAD':629}
PARTITION_FOLDS = {'train':tuple(range(1,9)), 'validation':(9,), 'test':(10,)}


def file_sha256(path):
    """Streaming integrity verification; never mutate a source/export file."""
    h=hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda:source.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class RhythmEntry:
    ecg_id: int
    patient_id: int
    strat_fold: int
    label: int
    rhythm_code: str
    file: str
    sha256: str


@dataclass(frozen=True)
class OfficialPartitions:
    train: tuple[int,...]
    validation: tuple[int,...]
    test: tuple[int,...]


class PTBXLExportIndex:
    """Validated, deterministic metadata index; all NPZ paths must exist.

    require_full_cohort=False is for small controlled tests only. By default the
    completed 19605-record export and Phase-12 metadata cohort must agree.
    Dataset rows are in ascending ecg_id order; no random split or sample dropping.
    """
    def __init__(self, root=DEFAULT_ROOT, manifest=DEFAULT_MANIFEST, *,
                 require_full_cohort=True, verify_source_metadata=True):
        self.root=Path(root).resolve()
        candidate=Path(manifest)
        self.manifest_path=candidate.resolve() if candidate.is_absolute() else (self.root/candidate).resolve()
        if not self.manifest_path.is_relative_to(self.root):
            raise ValueError('Manifest must belong to the selected versioned export directory')
        try:
            content=json.loads(self.manifest_path.read_text())
        except (OSError,ValueError) as exc:
            raise ValueError(f'Cannot read export manifest {self.manifest_path}: {exc}') from exc
        if not isinstance(content,dict):raise ValueError('Export manifest must be a JSON object')
        rows=content.get('records',[])
        if not isinstance(rows,list) or any(not isinstance(row,dict) for row in rows):
            raise ValueError('Manifest records must be a list of objects')
        if content.get('status')!='complete' or not rows:
            raise ValueError('A nonempty complete export manifest is required; incomplete historical runs are rejected')
        if any(content.get(k)!=0 for k in ('failed_records','not_attempted_records')):
            raise ValueError('Manifest contains failed or unattempted records')
        if content.get('selected_records')!=len(rows) or content.get('successful_records')!=len(rows):
            raise ValueError('Manifest success counts disagree with record entries')
        entries=[]
        for row in rows:
            try:
                eid,pid,fold=(row[k] for k in ('ecg_id','patient_id','strat_fold'))
                if any(isinstance(v,bool) or not isinstance(v,int) or v<1 for v in (eid,pid,fold)) or fold>10:
                    raise ValueError('Positive integer IDs and fold 1..10 required')
                code=row['label']; label=CLASS_NAMES.index(code)
                name=row['file']; digest=row['sha256']
                if row['status'] not in ('exported','validated_existing_export'):
                    raise ValueError('Unsuccessful record in a complete manifest')
                if name!=f'recordings/{eid:05d}_rhythm.npz':
                    raise ValueError('Unexpected record path or ECG identity')
                if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
                    raise ValueError('Invalid SHA256')
                path=(self.root/name).resolve()
                if not path.is_relative_to(self.root):raise ValueError('Record escapes dataset directory')
                if not path.is_file():raise FileNotFoundError(f'Missing export for ecg_id={eid}: {path}')
                entries.append(RhythmEntry(eid,pid,fold,label,code,name,digest))
            except (KeyError,TypeError,ValueError) as exc:
                raise ValueError(f'Invalid manifest record {row.get("ecg_id", "unknown")}: {exc}') from exc
        if len({e.ecg_id for e in entries})!=len(entries) or len({e.file for e in entries})!=len(entries):
            raise ValueError('Duplicate ECG identity or file path')
        if [e.ecg_id for e in entries]!=sorted(e.ecg_id for e in entries):
            raise ValueError('Export ordering must be ascending ecg_id')
        self.entries=tuple(entries)
        self.manifest=content
        self.manifest_sha256=file_sha256(self.manifest_path)
        counts=dict(Counter(e.rhythm_code for e in entries))
        patients={e.patient_id for e in entries}
        if content.get('successful_patients')!=len(patients):
            raise ValueError('Patient count disagrees with manifest')
        if require_full_cohort and (len(entries)!=19605 or len(patients)!=17223 or counts!=EXPECTED_COUNTS):
            raise ValueError('Completed cohort differs from verified Phase-12 counts')
        self.configuration=None
        if require_full_cohort:
            try:
                identity=json.loads((self.root/'dataset_identity.json').read_text())
            except (OSError,ValueError) as exc:
                raise ValueError('Cannot verify immutable dataset identity') from exc
            if identity!=content.get('dataset') or identity.get('configuration')!=preprocessing_config():
                raise ValueError('Dataset identity or Phase-13 preprocessing configuration differs')
            self.configuration=identity['configuration']
        self.source_metadata=None
        if verify_source_metadata:
            self.source_metadata=load_ptb_metadata()
            audit=propose_exclusive_cohort(self.source_metadata)['record_audit']
            expected={r['ecg_id']:r for r in audit if r['proposed_label'] in CLASS_NAMES}
            if require_full_cohort and set(expected)!={e.ecg_id for e in entries}:
                raise ValueError('Export cohort differs from original exclusive rhythm metadata')
            for e in entries:
                row=expected.get(e.ecg_id)
                if row is None or (e.patient_id,e.strat_fold,e.rhythm_code)!=(row['patient_id'],row['strat_fold'],row['proposed_label']):
                    raise ValueError(f'Original patient/fold/label metadata disagrees for ecg_id={e.ecg_id}')
        # Validate patient separation before any data loader or weights can be built.
        self.partitions=official_partitions(self)

    def __len__(self):return len(self.entries)

    def load_numpy(self, position, *, verify_checksum=True):
        """Load exactly one validated record; expose failures, never skip a row."""
        if isinstance(position,bool) or not isinstance(position,Integral) or not 0<=position<len(self):
            raise IndexError('Dataset index must be an integer in range')
        entry=self.entries[int(position)];path=self.root/entry.file
        try:
            if verify_checksum and file_sha256(path)!=entry.sha256:
                raise ValueError('SHA256 mismatch (corrupt or changed export)')
            arrays=load_pilot_record(path)
            for field in ('ecg_id','patient_id','strat_fold'):
                if int(arrays[field])!=getattr(entry,field):raise ValueError(f'{field} does not match manifest')
            if int(arrays['y'])!=entry.label or str(arrays['rhythm_code'])!=entry.rhythm_code:
                raise ValueError('Class label does not match manifest')
            if self.configuration is not None:
                config=json.loads(str(arrays['preprocessing_configuration']))
                fitted=config.pop('fitted_normalization')
                if config!=self.configuration or fitted['source_ids']!=[str(entry.ecg_id)]:
                    raise ValueError('Per-record normalization/preprocessing configuration differs')
            if self.source_metadata is not None:
                expected=record_identity(self.source_metadata,entry.ecg_id)
                if json.loads(str(arrays['original_metadata']))!=expected:
                    raise ValueError('Original SCP/patient metadata differs from source CSV')
            return arrays
        except (OSError,ValueError,KeyError,TypeError) as exc:
            raise ValueError(f'Cannot load PTB-XL ecg_id={entry.ecg_id} at {path}: {exc}') from exc


def official_partitions(index):
    """Use only original folds. No patient may occur in multiple official folds."""
    patients={}
    groups={name:[] for name in PARTITION_FOLDS}
    for i,entry in enumerate(index.entries):
        previous=patients.setdefault(entry.patient_id,entry.strat_fold)
        if previous!=entry.strat_fold:
            raise ValueError(f'Patient {entry.patient_id} occurs in more than one official fold')
        for name,folds in PARTITION_FOLDS.items():
            if entry.strat_fold in folds:groups[name].append(i)
    sets=[set(groups[name]) for name in PARTITION_FOLDS]
    if set.union(*sets)!=set(range(len(index.entries))) or any(sets[i]&sets[j] for i in range(3) for j in range(i)):
        raise ValueError('Official partitions must be disjoint and cover the whole cohort')
    patient_sets=[{index.entries[i].patient_id for i in groups[name]} for name in PARTITION_FOLDS]
    if any(patient_sets[i]&patient_sets[j] for i in range(3) for j in range(i)):
        raise ValueError('Patient leakage across fold groups')
    return OfficialPartitions(**{name:tuple(values) for name,values in groups.items()})


def training_class_counts(index, training_indices):
    """Count only explicitly supplied, unique indices belonging to folds 1..8."""
    indices=np.asarray(training_indices)
    if indices.ndim!=1 or indices.dtype.kind not in 'iu' or len(indices)==0:
        raise ValueError('Explicit nonempty integer training indices required')
    if len(np.unique(indices))!=len(indices) or np.any(indices<0) or np.any(indices>=len(index)):
        raise ValueError('Duplicate or out-of-range training indices')
    entries=[index.entries[int(i)] for i in indices]
    if any(e.strat_fold not in PARTITION_FOLDS['train'] for e in entries):
        raise ValueError('Validation/test labels must never be used to calculate training counts or weights')
    return np.bincount([e.label for e in entries],minlength=4).astype(np.int64)


def training_class_weights(index, training_indices):
    """Balanced CrossEntropyLoss weights n_train/(4*n_class), training data only.

    Return NumPy float32, converted to torch.float32 on the loss device by caller.
    Missing classes are an error, never infinity or fabricated counts.
    """
    counts=training_class_counts(index,training_indices)
    if np.any(counts==0):raise ValueError('All four classes must be present to calculate balanced weights')
    return (int(counts.sum())/(4*counts.astype(np.float64))).astype(np.float32)
