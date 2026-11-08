"""Descriptive imbalance analysis and opt-in utilities for an established train subset."""
from dataclasses import dataclass
from numbers import Integral
import numpy as np
from .data_loader import DEFAULT_DATA_DIR, list_recordings, load_recording
from .segmentation import segment_recording
from .label_mapping import CLASS_NAMES, LabeledHeartbeats, map_segments, class_distribution


def describe_counts(counts, *, rare_count_threshold=1000):
    """Describe all five classes; rare means 0 < count < threshold (configurable)."""
    if isinstance(rare_count_threshold, bool) or not isinstance(rare_count_threshold, Integral) or rare_count_threshold < 1:
        raise ValueError('rare_count_threshold must be a positive integer')
    if set(counts) != set(CLASS_NAMES):
        raise ValueError('Counts must contain exactly N, S, V, F, Q')
    if any(isinstance(v, bool) or not isinstance(v, Integral) or v < 0 for v in counts.values()):
        raise ValueError('Counts must be nonnegative integers')
    counts = {name:int(counts[name]) for name in CLASS_NAMES}
    total = sum(counts.values())
    present = [name for name in CLASS_NAMES if counts[name] > 0]
    largest = max((counts[name] for name in present), default=0)
    smallest = min((counts[name] for name in present), default=0)
    return {'class_counts':counts,'mapped_count':total,
        'class_percentages':{name:100*counts[name]/total if total else 0.0 for name in CLASS_NAMES},
        'majority_classes':[name for name in present if counts[name]==largest],
        'minority_nonempty_classes':[name for name in present if counts[name]==smallest],
        'majority_to_minority_nonempty_ratio':largest/smallest if smallest else None,
        'zero_classes':[name for name in CLASS_NAMES if counts[name]==0],
        'rare_classes':[name for name in present if counts[name]<rare_count_threshold],
        'rare_count_threshold':int(rare_count_threshold)}


def analyze_labeled(labeled, *, rare_count_threshold=1000):
    """Analyze existing labels without changing ECG or source metadata."""
    if not isinstance(labeled,LabeledHeartbeats):
        raise ValueError('Expected LabeledHeartbeats')
    if labeled.y.ndim != 1 or len(labeled.y) != len(labeled.X) or labeled.y.dtype.kind not in 'iu':
        raise ValueError('Invalid label array')
    if np.any(labeled.y < -1) or np.any(labeled.y >= len(CLASS_NAMES)):
        raise ValueError('Invalid label index')
    expected=np.array([CLASS_NAMES[i] if i>=0 else 'UNMAPPED' for i in labeled.y])
    if not np.array_equal(expected,labeled.mapped_classes):
        raise ValueError('Integer and named labels disagree')
    distribution=class_distribution(labeled)
    counts={name:int(np.count_nonzero(labeled.y==i)) for i,name in enumerate(CLASS_NAMES)}
    if counts != distribution['class_counts'] or sum(counts.values())+distribution['unmapped_count'] != len(labeled.X):
        raise ValueError('Labels and segmentation counts disagree')
    metadata=labeled.segments
    pairs=sorted(set(zip(metadata.record_ids.tolist(),metadata.patient_group_ids.tolist())))
    return {**distribution,**describe_counts(counts,rare_count_threshold=rare_count_threshold),
            'record_patient_groups':[{'record_id':rid,'patient_group_id':group or None} for rid,group in pairs],
            'preprocessing_configuration':metadata.configuration}


def analyze_database(root=DEFAULT_DATA_DIR, *, record_ids=None, rare_count_threshold=1000, progress=None):
    """Process actual complete local records one at a time; default is the full database.

    No ECG arrays are concatenated or exported, and no balancing is invoked.
    Unknown mapped labels stay explicit. Recording IDs never imply patient identity.
    """
    ids=list_recordings(root) if record_ids is None else list(record_ids)
    if len(ids)!=len(set(ids)):
        raise ValueError('Duplicate recording IDs')
    describe_counts(dict.fromkeys(CLASS_NAMES,0),rare_count_threshold=rare_count_threshold)
    records={}
    totals=dict.fromkeys(CLASS_NAMES,0)
    segment_total=unmapped_total=unresolved_total=annotations_total=nonbeat_total=boundary_total=0
    for rid in ids:
        record=load_recording(rid,root)
        labeled=map_segments(segment_recording(record),unknown_policy='mark')
        result=analyze_labeled(labeled,rare_count_threshold=rare_count_threshold)
        records[rid]=result
        for name in CLASS_NAMES:
            totals[name]+=result['class_counts'][name]
        segment_total+=result['segment_count']
        unmapped_total+=result['unmapped_count']
        unresolved_total+=result['patient_groups_unresolved_count']
        report=result['segmentation_report']
        if report['extracted_beats']!=result['segment_count'] or report['total_annotations'] != (
                result['segment_count']+report['excluded_nonheartbeat']+report['skipped_boundary']):
            raise ValueError(f'Segmentation audit failed for {rid}')
        annotations_total+=report['total_annotations']
        nonbeat_total+=report['excluded_nonheartbeat']
        boundary_total+=report['skipped_boundary']
        if progress is not None:
            progress(rid,result)
    aggregate=describe_counts(totals,rare_count_threshold=rare_count_threshold)
    assert aggregate['mapped_count']+unmapped_total==segment_total
    aggregate.update({'segment_count':segment_total,'unmapped_count':unmapped_total,
        'patient_groups_unresolved_count':unresolved_total,'total_annotations':annotations_total,
        'excluded_nonheartbeat':nonbeat_total,'skipped_boundary':boundary_total})
    return {'record_ids':ids,'record_count':len(ids),'class_names':list(CLASS_NAMES),
            'mapping':dict(labeled.mapping) if ids else {},'records':records,'aggregate':aggregate,
            'records_with_rare_classes':{rid:r['rare_classes'] for rid,r in records.items() if r['rare_classes']},
            'records_missing_classes':{rid:r['zero_classes'] for rid,r in records.items() if r['zero_classes']}}


@dataclass(frozen=True)
class TrainingSubset:
    """Explicit, caller-attested train selection; no split is created or verified here."""
    source_indices: np.ndarray
    y: np.ndarray
    record_ids: np.ndarray
    patient_group_ids: np.ndarray
    annotation_positions: np.ndarray
    partition: str
    patient_split_confirmed: bool
    source_size: int


def _validate_training(subset):
    if not isinstance(subset,TrainingSubset) or subset.partition!='train' or subset.patient_split_confirmed is not True:
        raise ValueError('An explicitly confirmed TrainingSubset is required')
    if isinstance(subset.source_size,bool) or not isinstance(subset.source_size,Integral) or subset.source_size<1:
        raise ValueError('Invalid source dataset size')
    n=len(subset.y)
    arrays=(subset.source_indices,subset.y,subset.record_ids,subset.patient_group_ids,subset.annotation_positions)
    if not n or any(np.asarray(a).ndim!=1 or len(a)!=n for a in arrays):
        raise ValueError('Training subset must be nonempty with aligned metadata')
    if subset.y.dtype.kind not in 'iu' or np.any(subset.y<0) or np.any(subset.y>=5):
        raise ValueError('Training labels must be mapped indices 0..4')
    indices=subset.source_indices
    if indices.dtype.kind not in 'iu' or np.any(indices<0) or np.any(indices>=subset.source_size) or len(np.unique(indices))!=n:
        raise ValueError('Invalid or duplicate source indices')
    if n>=subset.source_size:
        raise ValueError('The full source dataset cannot be used as a training subset')
    if any(not isinstance(g,str) or not g for g in subset.patient_group_ids):
        raise ValueError('Training patient groups must be resolved by the caller')


def make_training_subset(labeled, indices, *, partition, patient_split_confirmed):
    """Select supplied indices ONLY after Anna establishes patient-separated splits.

    Requires a proper subset and resolved patient groups. These checks cannot prove
    the correctness of the split or that the supplied source is the complete dataset.
    """
    if not isinstance(labeled,LabeledHeartbeats):
        raise ValueError('Expected LabeledHeartbeats')
    chosen=np.asarray(indices)
    if chosen.ndim!=1 or chosen.dtype.kind not in 'iu' or not chosen.size:
        raise ValueError('Explicit nonempty integer training indices are required')
    if np.any(chosen<0) or np.any(chosen>=len(labeled.y)):
        raise ValueError('Training index outside source data')
    source=labeled.segments
    result=TrainingSubset(chosen.astype(np.int64,copy=True),labeled.y[chosen].copy(),
        source.record_ids[chosen].copy(),source.patient_group_ids[chosen].copy(),
        source.annotation_positions[chosen].copy(),partition,patient_split_confirmed,len(labeled.y))
    _validate_training(result)
    for a in (result.source_indices,result.y,result.record_ids,result.patient_group_ids,result.annotation_positions):
        a.setflags(write=False)
    return result


def training_class_weights(subset):
    """Inverse-frequency weights n/(k_present * count); absent classes get zero.

    Uses only supplied training labels; zero weight does not teach a missing class.
    Returned keys are fixed class indices, with no model/loss implementation.
    """
    _validate_training(subset)
    counts=np.bincount(subset.y,minlength=5)
    present=np.count_nonzero(counts)
    return {i:float(len(subset.y)/(present*c)) if c else 0.0 for i,c in enumerate(counts)}


def training_sampling_indices(subset, *, strategy, random_seed):
    """Return source indices only, balancing PRESENT training classes.

    oversample: each present class reaches the largest training count, retaining
    originals and drawing extra indices with replacement. undersample: each reaches
    the smallest nonzero count, without replacement. No new ECG is synthesized.
    """
    _validate_training(subset)
    if strategy not in ('oversample','undersample'):
        raise ValueError('strategy must be oversample or undersample')
    if isinstance(random_seed,bool) or not isinstance(random_seed,Integral) or random_seed<0:
        raise ValueError('An explicit nonnegative integer random_seed is required')
    rng=np.random.default_rng(random_seed)
    classes=[np.flatnonzero(subset.y==i) for i in range(5) if np.any(subset.y==i)]
    target=(max if strategy=='oversample' else min)(len(a) for a in classes)
    selected=[]
    for rows in classes:
        if strategy=='oversample':
            rows=np.concatenate([rows,rng.choice(rows,size=target-len(rows),replace=True)])
        else:
            rows=rng.choice(rows,size=target,replace=False)
        selected.extend(subset.source_indices[rows].tolist())
    result=np.asarray(selected,dtype=np.int64)
    rng.shuffle(result)
    return result
