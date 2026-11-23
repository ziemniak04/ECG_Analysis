"""Label-free per-record timing features and training-only fitted transforms.

No ECG processing or R-peak detection. Inputs are validated heartbeat timestamps.
All intervals are seconds. Missing raw entries are zero plus explicit bool masks.
"""
from dataclasses import asdict
from pathlib import Path
import json
import numpy as np
from .mitbih_rr_config import RRFeatureConfig,RAW_RR_NAMES,RR_FEATURE_NAMES


def beat_ids(record_id,annotation_indices,positions):
    if not isinstance(record_id,str) or not record_id:raise ValueError('Explicit recording ID required')
    a=np.asarray(annotation_indices);p=np.asarray(positions)
    if a.ndim!=1 or p.shape!=a.shape or a.dtype.kind not in 'iu' or p.dtype.kind not in 'iu':
        raise ValueError('Aligned integer source annotation vectors required')
    if np.any(a<0) or np.any(p<0):raise ValueError('Nonnegative annotation identity required')
    ids=np.asarray([f'{record_id}:{int(i)}:{int(t)}' for i,t in zip(a,p)],dtype=str)
    if len(set(ids.tolist()))!=len(ids):raise ValueError('Duplicate stable beat IDs')
    return ids


def rr_features(heartbeat_positions,target_positions,sampling_frequency,config=None):
    """No symbol/class/ECG input. One call must describe exactly ONE recording.

    For beat k: previous=d[k]; baseline=median of up to eight d[j], j<k.
    Here d[k]=(p[k]-p[k-1])/fs, so the current previous interval is EXCLUDED
    from baseline. No next RR or future interval is used. Earliest valid
    heartbeat, even if its ECG window was skipped, supplies timing context.
    """
    config=RRFeatureConfig() if config is None else config
    if not isinstance(config,RRFeatureConfig):raise TypeError('RRFeatureConfig required')
    p=np.asarray(heartbeat_positions);t=np.asarray(target_positions)
    if p.ndim!=1 or t.ndim!=1 or p.dtype.kind not in 'iu' or t.dtype.kind not in 'iu':
        raise ValueError('Integer heartbeat/target positions required')
    if np.any(p<0) or np.any(t<0) or np.any(np.diff(p.astype(np.float64))<=0) or np.any(np.diff(t.astype(np.float64))<=0):
        raise ValueError('Strictly increasing nonnegative beat positions required; duplicates are not silently removed')
    if isinstance(sampling_frequency,bool) or not np.isscalar(sampling_frequency) or not np.isfinite(sampling_frequency) or sampling_frequency<=0:
        raise ValueError('Positive finite sampling frequency required')
    loc=np.searchsorted(p,t)
    if np.any(loc>=len(p)) or not np.array_equal(p[loc],t):raise ValueError('Every target must match a valid heartbeat')
    d=np.diff(p.astype(np.float64))/float(sampling_frequency)
    if not np.isfinite(d).all() or np.any(d<=0):raise ValueError('Invalid intervals')
    raw=np.zeros((len(t),3),dtype=np.float64);valid=np.zeros_like(raw,dtype=bool)
    for row,k in enumerate(loc):
        if k>=1:raw[row,0]=d[k-1];valid[row,0]=True
        prior=d[max(0,k-1-config.history_intervals):max(0,k-1)]
        if len(prior):raw[row,1]=np.median(prior);valid[row,1]=True
        if valid[row,0] and valid[row,1]:raw[row,2]=raw[row,0]/raw[row,1];valid[row,2]=True
    if not np.isfinite(raw).all():raise ValueError('RR numeric overflow')
    return raw,valid


def validate_raw(raw,valid):
    raw=np.asarray(raw);valid=np.asarray(valid)
    if raw.ndim!=2 or raw.shape[1]!=3 or valid.shape!=raw.shape or valid.dtype!=np.bool_:
        raise ValueError('Aligned [N,3] raw values and bool masks required')
    if not np.isfinite(raw).all() or np.any(raw[~valid]!=0) or np.any(raw[valid]<=0):
        raise ValueError('Finite positive observed values, zero missing sentinel required')
    if np.any(valid[:,1]&~valid[:,0]) or not np.array_equal(valid[:,2],valid[:,0]&valid[:,1]):
        raise ValueError('Inconsistent causal availability masks')
    return raw.astype(np.float64,copy=False),valid


def fit_training_transform(raw,valid,*,partition,patient_groups,expected_training_groups,manifest_sha256):
    """Caller must supply exact TRAIN membership; no y or class labels accepted.

    Fit median/mean/population std on observed TRAIN entries only. Validation
    values never influence these parameters. The exporter verifies membership
    and source identity before calling this primitive.
    """
    if partition!='train':raise PermissionError('Fit is allowed exclusively on TRAIN')
    raw,valid=validate_raw(raw,valid)
    groups=np.asarray(patient_groups)
    if groups.shape!=(len(raw),) or not len(raw) or set(groups.tolist())!=set(expected_training_groups):
        raise ValueError('Exact nonempty training-group coverage required')
    if not expected_training_groups or len(set(expected_training_groups))!=len(expected_training_groups):
        raise ValueError('Explicit unique training groups required')
    if not isinstance(manifest_sha256,str) or len(manifest_sha256)!=64:raise ValueError('Source manifest identity required')
    impute=[];mean=[];scale=[];counts=[]
    for col in range(3):
        observed=raw[valid[:,col],col]
        if not len(observed):raise ValueError('No observed training values for feature')
        sd=float(observed.std(ddof=0))
        impute.append(float(np.median(observed)));mean.append(float(observed.mean()))
        scale.append(sd if sd>1e-8 else 1.);counts.append(len(observed))
    return dict(schema='mitbih-rr-transform-v1',feature_names=list(RAW_RR_NAMES),
        fitted_partition='train',training_groups=sorted(expected_training_groups),training_rows=len(raw),
        source_manifest_sha256=manifest_sha256,imputation_median=impute,mean=mean,scale=scale,
        valid_training_counts=counts,standard_deviation_ddof=0,constant_scale_policy='std<=1e-8 -> scale 1',
        imputation_policy='training observed median; availability masks retained',clipping=False)


def transform_rr(raw,valid,state):
    raw,valid=validate_raw(raw,valid)
    if (state.get('schema')!='mitbih-rr-transform-v1' or state.get('fitted_partition')!='train'
            or state.get('feature_names')!=list(RAW_RR_NAMES)):raise ValueError('Wrong transform contract')
    median=np.asarray(state['imputation_median'],dtype=float);mean=np.asarray(state['mean'],dtype=float)
    scale=np.asarray(state['scale'],dtype=float)
    if any(v.shape!=(3,) or not np.isfinite(v).all() for v in (median,mean,scale)) or np.any(scale<=0):
        raise ValueError('Invalid stored transform parameters')
    values=(np.where(valid,raw,median)-mean)/scale
    result=np.concatenate((values,valid.astype(float)),axis=1).astype(np.float32)
    if not np.isfinite(result).all():raise ValueError('Transformed RR overflow')
    return result
