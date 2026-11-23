"""Five beat-group metrics; reuse the tested tied-score AP implementation."""
import numpy as np
from .label_mapping import CLASS_NAMES
from .ptbxl_metrics import binary_average_precision


def classification_metrics(labels, probabilities):
    """Rows=true, columns=argmax. Undefined metrics are None, not invented zero.

    Macro metrics require all five supported components. AP is non-interpolated
    one-vs-rest average precision, not trapezoidal PR area. Tied scores are grouped.
    """
    y = np.asarray(labels)
    p = np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or not len(y) or y.dtype.kind not in 'iu' or np.any((y < 0) | (y > 4)):
        raise ValueError('Nonempty integer beat labels 0..4 required')
    if p.shape != (len(y), 5) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Finite [N,5] probabilities required')
    if not np.allclose(p.sum(axis=1), 1, atol=1e-6, rtol=1e-6):
        raise ValueError('Probability rows must sum to one')
    cm = np.zeros((5, 5), dtype=np.int64)
    np.add.at(cm, (y, p.argmax(axis=1)), 1)
    rows = []
    for i, name in enumerate(CLASS_NAMES):
        tp = int(cm[i, i]); fn = int(cm[i].sum()) - tp; fp = int(cm[:, i].sum()) - tp
        tn = len(y) - tp - fn - fp; support = tp + fn
        rows.append(dict(class_name=name, class_index=i, support=support, tp=tp, fp=fp, fn=fn, tn=tn,
            precision=tp/(tp+fp) if tp+fp else None, recall=tp/support if support else None,
            specificity=tn/(tn+fp) if tn+fp else None,
            f1=2*tp/(2*tp+fp+fn) if support else None,
            auprc=binary_average_precision((y == i).astype(int), p[:, i])))
    def macro(key):
        values = [r[key] for r in rows]
        return float(np.mean(values)) if all(v is not None for v in values) else None
    return dict(beat_count=len(y), accuracy=float(np.trace(cm)/len(y)), macro_f1=macro('f1'),
        macro_auprc=macro('auprc'), per_class=rows, confusion_matrix=cm.tolist(),
        absent_true_classes=[r['class_name'] for r in rows if not r['support']],
        confusion_matrix_axes='rows=true, columns=predicted; N,S,V,F,Q',
        auprc_definition='Non-interpolated one-vs-rest average precision; tied scores grouped')
