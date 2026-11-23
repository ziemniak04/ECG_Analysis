"""Validation-only SBRAD decision study. No fitting, training or probability calibration.

ABSTAIN is -1, never a fifth disease label. A rejected SBRAD winner is not
reassigned. Full-cohort recall counts abstained positives as missed detections.
"""
from numbers import Real
import math
import numpy as np
import torch
from torch.utils.data import DataLoader
from .ptbxl_metrics import CLASS_NAMES
from .ptbxl_protocol import PTBXLExportIndex
from .ptbxl_torch import PTBXLRhythmDataset

ABSTAIN = -1
THRESHOLDS = (0., .4, .45, .5, .55, .6, .65, .7, .75, .8, .85, .9, .95, .975, .99, 1.)
SELECTION = {'minimum_retained_baseline_true_positives': .90,
             'minimum_overall_coverage': .97,
             'objective': 'maximize full-cohort SBRAD F1; tie: highest coverage, then lowest threshold',
             'fallback': 'original argmax if no eligible improvement in SBRAD F1'}


def _probabilities(value):
    prob = np.asarray(value, dtype=float)
    if (prob.ndim != 2 or prob.shape[1] != 4 or not len(prob)
            or not np.isfinite(prob).all() or np.any(prob < 0) or np.any(prob > 1)
            or not np.allclose(prob.sum(axis=1), 1, atol=1e-6, rtol=1e-6)):
        raise ValueError('Nonempty finite [N,4] probability rows summing to one required')
    return prob


def sbrad_decisions(probabilities, threshold=0.):
    """Keep argmax; reject SBRAD winners with p(SBRAD) < threshold.

    Equality passes. NumPy's lowest-index argmax tie rule matches the baseline.
    A high SBRAD probability alone never overrides another winning class.
    """
    if isinstance(threshold, bool) or not isinstance(threshold, Real) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Finite threshold in [0,1] required')
    prob = _probabilities(probabilities)
    predicted = prob.argmax(axis=1).astype(np.int64)
    predicted[(predicted == 3) & (prob[:, 3] < threshold)] = ABSTAIN
    return predicted


def decision_metrics(labels, decisions):
    """One-vs-rest counts over ALL cases, plus explicitly conditional metrics.

    Abstention is a non-detection for each class: a true positive class that is
    rejected contributes FN, never vanishes from full-cohort recall. Non-alert
    negatives contribute TN. Conditional metrics use only accepted records.
    Macro F1 is undefined if any true class is absent from the respective cohort.
    """
    y, pred = np.asarray(labels), np.asarray(decisions)
    if (y.ndim != 1 or not len(y) or y.dtype.kind not in 'iu'
            or pred.shape != y.shape or pred.dtype.kind not in 'iu'
            or not np.isin(y, range(4)).all() or not np.isin(pred, [-1, 0, 1, 2, 3]).all()):
        raise ValueError('Aligned nonempty integer labels 0..3 and decisions -1..3 required')
    accepted = pred != ABSTAIN
    cm = np.zeros((4, 5), dtype=np.int64)
    np.add.at(cm, (y, np.where(accepted, pred, 4)), 1)
    per_class = []
    for i, name in enumerate(CLASS_NAMES):
        tp = int(((y == i) & (pred == i)).sum())
        fp = int(((y != i) & (pred == i)).sum())
        fn = int(((y == i) & (pred != i)).sum())
        tn = len(y) - tp - fp - fn
        support = int((y == i).sum()); abstained = int(((y == i) & ~accepted).sum())
        per_class.append({'class': name, 'class_index': i, 'support': support,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'precision': tp / (tp + fp) if tp + fp else None,
            'recall': tp / support if support else None,
            'specificity': tn / (tn + fp) if tn + fp else None,
            'f1': 2 * tp / (2 * tp + fp + fn) if support else None,
            'abstained': abstained, 'abstention_rate': abstained / support if support else None})
    n = int(accepted.sum())
    conditional_f1 = []
    for i in range(4):
        support = int(((y == i) & accepted).sum())
        tp = int(((y == i) & (pred == i)).sum())
        fp = int(((y != i) & (pred == i)).sum())
        fn = support - tp
        conditional_f1.append(2 * tp / (2 * tp + fp + fn) if support else None)
    def macro(values):
        return float(np.mean(values)) if all(v is not None for v in values) else None
    return {'record_count': len(y), 'accepted': n, 'abstained': len(y) - n,
            'coverage': n / len(y), 'abstention_rate': 1 - n / len(y),
            'accepted_accuracy_coverage_conditional': float((y[accepted] == pred[accepted]).mean()) if n else None,
            'accepted_macro_f1_coverage_conditional': macro(conditional_f1),
            'accepted_true_support': np.bincount(y[accepted], minlength=4).tolist(),
            'full_cohort_macro_f1_abstentions_as_misses': macro([r['f1'] for r in per_class]),
            'per_class': per_class, 'confusion_matrix_with_abstain': cm.tolist()}


def threshold_study(labels, probabilities, thresholds=THRESHOLDS):
    """Deterministic prespecified grid; zero is the unchanged argmax baseline."""
    rows = []
    for threshold in thresholds:
        metrics = decision_metrics(labels, sbrad_decisions(probabilities, threshold))
        metrics['threshold'] = float(threshold)
        rows.append(metrics)
    if not rows or rows[0]['threshold'] != 0 or len({r['threshold'] for r in rows}) != len(rows):
        raise ValueError('Unique threshold grid starting with zero required')
    return rows


def recommend_threshold(rows):
    """Engineering selection rule fixed before viewing threshold results.

    Retain >=90% of baseline TP and >=97% overall coverage. Choose maximum
    full-cohort SBRAD F1, then coverage and lowest threshold. No improvement:
    keep argmax. None of these constraints is a medical acceptance criterion.
    """
    if not rows or rows[0]['threshold'] != 0:
        raise ValueError('Argmax baseline must be first')
    baseline = rows[0]['per_class'][3]
    if baseline['tp'] <= 0 or baseline['f1'] is None:
        raise ValueError('Baseline must have true SBRAD detections')
    minimum_tp = math.ceil(SELECTION['minimum_retained_baseline_true_positives'] * baseline['tp'])
    eligible = [r for r in rows if r['per_class'][3]['tp'] >= minimum_tp
                and r['coverage'] >= SELECTION['minimum_overall_coverage']]
    selected = max(eligible, key=lambda r: (r['per_class'][3]['f1'], r['coverage'], -r['threshold']))
    if selected['per_class'][3]['f1'] <= baseline['f1']:
        selected = rows[0]
    return {'selected_threshold': selected['threshold'], 'minimum_sbrad_tp': minimum_tp,
            'eligible_thresholds': [r['threshold'] for r in eligible], 'criterion': SELECTION.copy()}


class ValidationOnlyIndex(PTBXLExportIndex):
    """Reuse the validated reader; disallow any non-fold-9 waveform before I/O."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.read_ecg_ids = []
        self.blocked_reads = 0

    def load_numpy(self, position, *, verify_checksum=True):
        if self.entries[position].strat_fold != 9:
            self.blocked_reads += 1
            raise RuntimeError('Only validation fold 9 waveforms are permitted')
        arrays = super().load_numpy(position, verify_checksum=verify_checksum)
        self.read_ecg_ids.append(self.entries[position].ecg_id)
        return arrays


def frozen_validation_predictions(model, index, batch_size=16):
    """Eval-only forward pass; assert state_dict (including BN buffers) unchanged."""
    if not isinstance(index, ValidationOnlyIndex):
        raise TypeError('A validation-only guarded index is required')
    before = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    loader = DataLoader(PTBXLRhythmDataset(index, partition='validation'), batch_size=batch_size,
                        shuffle=False, num_workers=0, drop_last=False)
    labels, probabilities, ids, patients, folds = [], [], [], [], []
    with torch.inference_mode():
        for batch in loader:
            if not bool((batch['strat_fold'] == 9).all()):
                raise RuntimeError('Non-validation record in inference batch')
            logits = model(batch['X'])
            if tuple(logits.shape) != (len(batch['y']), 4) or not bool(torch.isfinite(logits).all()):
                raise ValueError('Invalid frozen-model logits')
            probabilities.extend(torch.softmax(logits, dim=1).cpu().tolist())
            labels.extend(batch['y'].tolist()); ids.extend(batch['ecg_id'].tolist())
            patients.extend(batch['patient_id'].tolist()); folds.extend(batch['strat_fold'].tolist())
            if len(ids) % 320 == 0:
                print(f'Validation inference: {len(ids)}/{len(loader.dataset)}', flush=True)
    if not ids or len(ids) != len(loader.dataset) or len(set(ids)) != len(ids):
        raise RuntimeError('Missing or duplicate validation predictions')
    if any(not torch.equal(value.detach().cpu(), before[key]) for key, value in model.state_dict().items()):
        raise RuntimeError('Frozen model parameters or buffers changed')
    return {'labels': np.asarray(labels, dtype=np.int64),
            'probabilities': np.asarray(probabilities, dtype=np.float64),
            'ecg_id': np.asarray(ids, dtype=np.int64),
            'patient_id': np.asarray(patients, dtype=np.int64),
            'strat_fold': np.asarray(folds, dtype=np.int64)}
