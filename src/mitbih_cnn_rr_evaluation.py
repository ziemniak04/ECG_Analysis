"""Frozen validation-only CNN+RR comparison. No training or held-out loader."""
import json,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .mitbih_execution_guard import require_guard
from .mitbih_rr import transform_rr
from .mitbih_rr_dataset import collate_rr
from .models.mitbih_cnn_rr import MITBIHBeatCNNRR
from .mitbih_metrics import classification_metrics
from .mitbih_unweighted import assert_validation_batch,loss_summary
from .prepare_dataset import sha256_file


def neutral_rr(state,n):
    """All history unavailable: TRAIN median imputation, fixed scaling, zero masks.

    Not literal zeros after scaling, not a new morphology-only trained model.
    """
    if isinstance(n,bool) or not isinstance(n,int) or n<1:raise ValueError('Positive batch size required')
    return torch.from_numpy(transform_rr(np.zeros((n,3)),np.zeros((n,3),dtype=bool),state))


def evaluate(checkpoint,dataset,split,weights,*,neutral=False):
    require_guard()
    if dataset.partition!='validation':raise PermissionError('Only full validation is permitted')
    expected=set(split['summary']['validation']['record_ids'])
    if {e.record_id for e in dataset.base.shards}!=expected:raise ValueError('Incomplete validation cohort')
    path=Path(checkpoint);digest=sha256_file(path)
    if digest!=path.with_suffix('.sha256').read_text().strip():raise ValueError('Checkpoint checksum differs')
    saved=torch.load(path,map_location='cpu',weights_only=True)
    model=MITBIHBeatCNNRR()
    if saved.get('schema')!='mitbih-cnn-rr-best-model-v1' or saved.get('class_names')!=list('NSVFQ') or saved.get('model_configuration')!=model.configuration():raise ValueError('Checkpoint architecture/classes differ')
    if saved.get('rr_manifest_sha256')!=sha256_file(dataset.root/'manifest.json'):raise ValueError('RR source differs')
    if saved['preprocessing_configuration']['RR_transform']!=dataset.transform:raise ValueError('RR transform differs')
    model.load_state_dict(saved['model_state_dict']);model.eval()
    before={k:v.clone() for k,v in model.state_dict().items()}
    loader=DataLoader(dataset,batch_size=32,shuffle=False,num_workers=0,collate_fn=collate_rr)
    probabilities=[];labels=[];metadata=[];losses=[];forward=0.;started=time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            assert_validation_batch(batch,split)
            rr=neutral_rr(dataset.transform,len(batch['y'])) if neutral else batch['RR']
            tick=time.perf_counter();logits=model(batch['X'],rr);forward+=time.perf_counter()-tick
            if logits.shape!=(len(rr),5) or not torch.isfinite(logits).all():raise ValueError('Invalid logits')
            probabilities.extend(torch.softmax(logits,1).tolist());labels.extend(batch['y'].tolist())
            losses.extend(torch.nn.functional.cross_entropy(logits,batch['y'],reduction='none').tolist())
            for j in range(len(rr)):
                metadata.append(dict(record_id=batch['record_id'][j],patient_group=batch['patient_group'][j],original_symbol=batch['original_symbol'][j],annotation_position=int(batch['annotation_position'][j]),source_index=int(batch['global_source_index'][j]),beat_id=batch['beat_id'][j],channel_0=batch['channel_names'][j][0],channel_1=batch['channel_names'][j][1]))
    if len(labels)!=split['summary']['validation']['beats'] or len({r['source_index'] for r in metadata})!=len(labels):raise ValueError('Missing/duplicate validation rows')
    if not all(torch.equal(before[k],v) for k,v in model.state_dict().items()) or sha256_file(path)!=digest:raise ValueError('Parameters changed')
    y=np.asarray(labels,dtype=np.int64);p=np.asarray(probabilities)
    metrics=classification_metrics(y,p)
    metrics.update(checkpoint_sha256=digest,parameters_unchanged=True,final_test_read=False,RR_condition='all_missing_train_median_plus_zero_masks' if neutral else 'informative_frozen_sidecar',forward_seconds=forward,inference_ms_per_beat=1000*forward/len(y),total_pass_seconds=time.perf_counter()-started,sqrt_weighted_validation_ce=loss_summary(losses,y,weights),unweighted_validation_ce=loss_summary(losses,y,None))
    groups={}
    for group in sorted({r['patient_group'] for r in metadata}):
        mask=np.asarray([r['patient_group']==group for r in metadata]);groups[group]=classification_metrics(y[mask],p[mask])
    return metrics,groups,metadata,y,p,np.asarray(losses)


def verify_alignment(reference,candidate):
    """Stable source identities, not just equal counts or labels."""
    if len(reference)!=len(candidate):raise ValueError('Population lengths differ')
    keys=('record_id','patient_group','original_symbol','annotation_position','source_index','channel_0','channel_1')
    if any(any(a[k]!=b[k] for k in keys) for a,b in zip(reference,candidate)):raise ValueError('Validation source alignment differs')
    return True
