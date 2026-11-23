"""Validation diagnostics for selective F decisions; never remap rejected beats."""
import numpy as np
from .mitbih_metrics import classification_metrics

F_INDEX=3
UNCERTAIN=-1
THRESHOLDS=(0.,.3,.4,.5,.6,.7,.8,.9,.95)


def f_decisions(probabilities,threshold):
    p=np.asarray(probabilities,dtype=float)
    if p.ndim!=2 or p.shape[1]!=5 or not np.isfinite(p).all() or np.any((p<0)|(p>1)) or not np.allclose(p.sum(1),1,atol=1e-6,rtol=1e-6):
        raise ValueError('Finite normalized [N,5] scores required')
    if isinstance(threshold,bool) or not np.isscalar(threshold) or not np.isfinite(threshold) or not 0<=threshold<=1:raise ValueError('Threshold in [0,1] required')
    pred=p.argmax(1);result=pred.copy()
    result[(pred==F_INDEX)&(p[:,F_INDEX]<threshold)]=UNCERTAIN
    return result


def threshold_metrics(y,p,threshold):
    # Validate labels and probabilities with the original full-coverage metric contract.
    full=classification_metrics(y,p);y=np.asarray(y);p=np.asarray(p)
    pred=f_decisions(p,threshold);accept=pred!=UNCERTAIN
    tp=int(((y==3)&(pred==3)).sum());fp=int(((y!=3)&(pred==3)).sum());support=int((y==3).sum())
    accepted=classification_metrics(y[accept],p[accept]) if accept.any() else None
    row=dict(threshold=float(threshold),rule='argmax' if threshold==0 else 'F_argmax_and_threshold_else_UNCERTAIN',total=len(y),F_support=support,F_tp=tp,F_fp=fp,F_fn=support-tp,F_precision=tp/(tp+fp) if tp+fp else None,F_recall=tp/support if support else None,abstentions=int((~accept).sum()),coverage=float(accept.mean()),accepted_count=int(accept.sum()),accepted_only_accuracy=None if accepted is None else accepted['accuracy'],accepted_only_macro_f1=None if accepted is None else accepted['macro_f1'],accepted_absent_classes=[] if accepted is None else accepted['absent_true_classes'],full_coverage_argmax_macro_f1=full['macro_f1'])
    for i,c in enumerate('NSVFQ'):
        n=int((y==i).sum());a=int(((y==i)&~accept).sum())
        row[c+'_abstentions']=a;row[c+'_abstention_rate']=a/n if n else None
    return row


def score_summary(values):
    a=np.asarray(values,dtype=float)
    if a.ndim!=1 or not np.isfinite(a).all():raise ValueError('Finite score vector required')
    q=np.quantile(a,[0,.25,.5,.75,1]).tolist() if len(a) else [None]*5
    return dict(count=len(a),mean=float(a.mean()) if len(a) else None,minimum=q[0],q25=q[1],median=q[2],q75=q[3],maximum=q[4])
