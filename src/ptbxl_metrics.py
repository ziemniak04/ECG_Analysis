"""Fixed-label ECG metrics. Undefined values are None (JSON null), never fabricated.

AUPRC is non-interpolated average precision, with tied probability scores grouped
at one threshold. No sklearn dependency. Confusion rows=true, columns=predicted.
"""
import numpy as np

CLASS_NAMES=('SR','AFIB','STACH','SBRAD')


def binary_average_precision(target,scores):
    """Sum precision_at_threshold * delta_recall; require positives and negatives."""
    target=np.asarray(target);scores=np.asarray(scores,dtype=float)
    if target.ndim!=1 or scores.shape!=target.shape or not len(target):
        raise ValueError('Aligned nonempty binary target and score vectors required')
    if not np.isin(target,[0,1]).all() or not np.isfinite(scores).all():
        raise ValueError('Invalid binary targets or nonfinite scores')
    positives=int(target.sum())
    if positives==0 or positives==len(target):return None
    order=np.argsort(-scores,kind='stable');y=target[order];ranked=scores[order]
    ends=np.r_[np.flatnonzero(np.diff(ranked)!=0),len(y)-1]
    tp=np.cumsum(y,dtype=np.int64)[ends]
    precision=tp/(ends+1)
    delta_tp=np.diff(np.r_[0,tp])
    return float(np.sum(precision*delta_tp/positives))


def classification_metrics(y_true,probabilities):
    """Argmax decisions; full four-class macro is null if any component undefined.

    F1 is 2TP/(2TP+FP+FN) for a supported class (zero if missed completely).
    Unsupported classes have null recall/F1/AP; precision with no predictions is
    null. Specificity is null if there are no true negatives available.
    """
    y=np.asarray(y_true);prob=np.asarray(probabilities,dtype=float)
    if y.ndim!=1 or y.dtype.kind not in 'iu' or not len(y) or np.any(y<0) or np.any(y>=4):
        raise ValueError('Nonempty integer labels 0..3 required')
    if prob.shape!=(len(y),4) or not np.isfinite(prob).all() or np.any(prob<0) or np.any(prob>1):
        raise ValueError('Finite [N,4] probabilities in [0,1] required')
    if not np.allclose(prob.sum(axis=1),1,atol=1e-6,rtol=1e-6):
        raise ValueError('Probability rows must sum to one')
    predicted=prob.argmax(axis=1);cm=np.zeros((4,4),dtype=np.int64)
    np.add.at(cm,(y,predicted),1)
    rows=[]
    for i,name in enumerate(CLASS_NAMES):
        tp=int(cm[i,i]);fn=int(cm[i].sum())-tp;fp=int(cm[:,i].sum())-tp
        tn=len(y)-tp-fn-fp;support=tp+fn;negatives=tn+fp
        rows.append({'class':name,'class_index':i,'support':support,'predicted_count':tp+fp,
            'tp':tp,'fn':fn,'fp':fp,'tn':tn,
            'recall':tp/support if support else None,
            'specificity':tn/negatives if negatives else None,
            'precision':tp/(tp+fp) if tp+fp else None,
            'f1':2*tp/(2*tp+fp+fn) if support else None,
            'auprc':binary_average_precision((y==i).astype(int),prob[:,i])})
    f1=[r['f1'] for r in rows];ap=[r['auprc'] for r in rows]
    def macro(values):return float(np.mean(values)) if all(v is not None for v in values) else None
    def available(values):
        valid=[v for v in values if v is not None]
        return float(np.mean(valid)) if valid else None
    return {'record_count':len(y),'accuracy':float(np.trace(cm)/len(y)),
        'macro_f1':macro(f1),'macro_auprc':macro(ap),
        'macro_f1_over_available_classes':available(f1),'macro_auprc_over_available_classes':available(ap),
        'absent_true_classes':[r['class'] for r in rows if r['support']==0],
        'classes_with_undefined_auprc':[r['class'] for r in rows if r['auprc'] is None],
        'per_class':rows,'confusion_matrix':cm.tolist(),
        'confusion_matrix_axes':'rows=true, columns=predicted; SR,AFIB,STACH,SBRAD',
        'auprc_definition':'Non-interpolated average precision, one-vs-rest; tied scores grouped'}
