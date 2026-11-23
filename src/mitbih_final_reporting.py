"""Final metrics from saved predictions only. No original-shard access or model calls."""
from pathlib import Path
import json
import numpy as np
from .mitbih_metrics import classification_metrics


def collect_saved_predictions(out,protocol):
    columns={}
    for rid in protocol['summary']['test']['record_ids']:
        with np.load(Path(out)/'predictions'/f'{rid}_predictions.npz',allow_pickle=False) as z:
            data={k:z[k] for k in z.files}
        n=len(data['y'])
        if data['probabilities'].shape!=(n,5) or not np.all(data['record_ids']==rid):raise ValueError('Invalid saved prediction identity/shape')
        if not np.array_equal(data['predicted_indices'],data['probabilities'].argmax(1)):raise ValueError('Saved decisions differ from frozen argmax')
        for key,value in data.items():
            if len(value)!=n:raise ValueError('Unaligned saved prediction columns')
            columns.setdefault(key,[]).append(value)
    a={k:np.concatenate(v) for k,v in columns.items()}
    if len(a['y'])!=protocol['summary']['test']['beats'] or len(set(a['beat_ids']))!=len(a['y']) or len(set(a['global_source_indices']))!=len(a['y']):raise ValueError('Incomplete/duplicate saved test examples')
    if set(a['patient_groups'])!=set(protocol['split']['groups']['test']):raise ValueError('Wrong saved patient groups')
    np.testing.assert_array_equal(np.bincount(a['y'],minlength=5),[protocol['summary']['test']['class_counts'][c] for c in 'NSVFQ'])
    return a


def metrics_from_saved(a):
    metric=classification_metrics(a['y'],a['probabilities'])
    groups={g:classification_metrics(a['y'][a['patient_groups']==g],a['probabilities'][a['patient_groups']==g]) for g in sorted(set(a['patient_groups']))}
    return metric,groups
