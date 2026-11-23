"""Single loss-only controlled experiment and validation-only prediction audit."""
from collections import Counter
from dataclasses import asdict, replace
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from .mitbih_protocol import MITBIHExportIndex, load_split
from .mitbih_readiness import verify_frozen_protocol, resources
from .mitbih_training import MITBIHTrainingConfig, train_model, build_plan, fingerprint, loss_weights
from .mitbih_torch import MITBIHBeatDataset, collate_mitbih
from .mitbih_metrics import classification_metrics
from .mitbih_full_run import snapshot, csv_new
from .models.mitbih_cnn import MITBIHBeatCNN
from .prepare_dataset import sha256_file
from .ptbxl_training import write_json_new

BASELINE = Path('results/model_training/phase22_mitbih_full/run_001')
BASELINE_SHA256 = '83f914122bf389fe0ddde1abe229cafa81abf1e8f8973dbd03cfc86eacb00d75'


def controlled_config(baseline, candidate):
    """Reject every intended experimental change except removal of class weights."""
    expected = dict(baseline); expected['loss_strategy'] = 'unweighted'
    if baseline.get('loss_strategy') != 'sqrt_balanced' or asdict(candidate) != expected:
        raise ValueError('Only loss_strategy may differ from the frozen Phase-22 configuration')


def run(output, audit):
    output, audit = Path(output), Path(audit)
    if output.exists() or audit.exists(): raise FileExistsError('Use new run and audit directories')
    if sha256_file(BASELINE/'best_model.pt') != BASELINE_SHA256:
        raise ValueError('Phase-22 baseline checksum differs')
    baseline_config = json.loads((BASELINE/'configuration.json').read_text())
    config = replace(MITBIHTrainingConfig(**baseline_config), loss_strategy='unweighted')
    controlled_config(baseline_config, config)
    index = MITBIHExportIndex(); split = load_split(index); summary = verify_frozen_protocol(index, split)
    baseline_plan = json.loads((BASELINE/'training_plan.json').read_text())
    plan = build_plan(index, split, config)
    for key in ('manifest_sha256','split_sha256','split','summary','model_configuration','code_sha256','runtime_versions'):
        if plan[key] != baseline_plan[key]: raise ValueError(f'Controlled source/population difference: {key}')
    capacity = resources()
    if capacity['MemAvailable_bytes'] < 2*2**30 or capacity['disk_free_bytes'] < 128*2**20:
        raise RuntimeError('Insufficient resource headroom')
    audit.mkdir(parents=True,exist_ok=False); snapshot(audit)
    rows = []; symbol_rows = []
    for partition in ('train','validation'):
        for rid in summary[partition]['record_ids']:
            arrays = index.load_shard(rid,split,partition)
            counts = Counter(map(str,arrays['original_symbols']))
            for symbol,count in sorted(counts.items()):
                symbol_rows.append(dict(partition=partition,record_id=rid,patient_group=index.by_record[rid].patient_group,
                    original_symbol=symbol,count=count))
            rows.append(dict(partition=partition,record_id=rid,patient_group=index.by_record[rid].patient_group,
                channel_0=str(arrays['channel_names'][0]),channel_1=str(arrays['channel_names'][1]),
                **dict(zip('NSVFQ',map(int,np.bincount(arrays['y'],minlength=5))))))
    csv_new(audit/'record_lead_class_distribution.csv',rows)
    csv_new(audit/'original_symbol_distribution.csv',symbol_rows)
    write_json_new(audit/'preflight.json',dict(status='passed',configuration=asdict(config),
        only_experimental_change='sqrt_balanced -> unweighted CE',initialization='Fresh seed=42; no checkpoint initialization',
        baseline_sha256=BASELINE_SHA256, resources=capacity, summary=summary, final_test_read=False,
        source_manifest_sha256=index.manifest_sha256,training_plan_sha256=fingerprint(plan)))
    print(json.dumps({'preflight':'passed','configuration':asdict(config),'output':str(output)},indent=2),flush=True)
    started=time.perf_counter()
    result=train_model(config,output,index=index,split=split,allow_full_training=True)
    allowed=set(summary['train']['record_ids']+summary['validation']['record_ids'])
    if not set(index.read_records)<=allowed:raise RuntimeError('Final-test read')
    write_json_new(output/'execution.json',dict(wall_seconds=time.perf_counter()-started,result=result,
        resources_after=resources(),read_records_including_preflight=index.read_records,final_test_read=False))
    return result


def assert_validation_batch(batch, split):
    if not set(batch['patient_group']) <= set(split['groups']['validation']):
        raise PermissionError('Only validation groups may be scored')
    if not set(batch['record_id']) <= set(split['summary']['validation']['record_ids']):
        raise PermissionError('Non-validation recording')


def loss_summary(losses, labels, weights):
    """CE aggregated exactly as PyTorch weighted mean: sum(w_y*CE)/sum(w_y)."""
    ce=np.asarray(losses,dtype=float);y=np.asarray(labels)
    if ce.ndim!=1 or ce.shape!=y.shape or not len(y) or not np.isfinite(ce).all():
        raise ValueError('Finite aligned losses and labels required')
    if y.dtype.kind not in 'iu' or np.any((y<0)|(y>4)):raise ValueError('Labels 0..4 required')
    w=np.ones(5) if weights is None else np.asarray(weights,dtype=float)
    if w.shape!=(5,) or not np.isfinite(w).all() or np.any(w<=0):raise ValueError('Positive class weights required')
    return dict(numerator=float(np.dot(ce,w[y])),denominator=float(w[y].sum()),
                loss=float(np.dot(ce,w[y])/w[y].sum()))


def evaluate(checkpoint, dataset, split, original_weights):
    """One frozen full-validation pass. No test Dataset, updates or thresholds."""
    if dataset.partition!='validation':raise PermissionError('Only full validation is permitted')
    expected = set(split['summary']['validation']['record_ids'])
    if {e.record_id for e in dataset.shards}!=expected:raise ValueError('Incomplete validation cohort')
    digest=sha256_file(checkpoint)
    if digest!=Path(checkpoint).with_suffix('.sha256').read_text().strip():raise ValueError('Checkpoint checksum differs')
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if saved.get('schema')!='mitbih-best-model-v1' or saved.get('class_names')!=list('NSVFQ'):
        raise ValueError('Wrong checkpoint type/classes')
    model=MITBIHBeatCNN();model.load_state_dict(saved['model_state_dict']);model.eval()
    before={k:v.clone() for k,v in model.state_dict().items()}
    loader=DataLoader(dataset,batch_size=32,shuffle=False,num_workers=0,collate_fn=collate_mitbih)
    probabilities=[];labels=[];losses=[];metadata=[];forward=0.;started=time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            assert_validation_batch(batch,split)
            tick=time.perf_counter();logits=model(batch['X']);forward+=time.perf_counter()-tick
            if not bool(torch.isfinite(logits).all()):raise ValueError('Invalid logits')
            ce=torch.nn.functional.cross_entropy(logits,batch['y'],reduction='none')
            probabilities.extend(torch.softmax(logits,1).tolist());labels.extend(batch['y'].tolist());losses.extend(ce.tolist())
            for j in range(len(batch['y'])):
                metadata.append(dict(record_id=batch['record_id'][j],patient_group=batch['patient_group'][j],
                    original_symbol=batch['original_symbol'][j],annotation_position=int(batch['annotation_position'][j]),
                    source_index=int(batch['global_source_index'][j]),channel_0=batch['channel_names'][j][0],
                    channel_1=batch['channel_names'][j][1]))
    if len(labels)!=split['summary']['validation']['beats'] or len({r['source_index'] for r in metadata})!=len(labels):
        raise ValueError('Missing/duplicate validation examples')
    if not all(torch.equal(before[k],v) for k,v in model.state_dict().items()) or sha256_file(checkpoint)!=digest:
        raise ValueError('Model changed during inference')
    y=np.asarray(labels,dtype=np.int64);prob=np.asarray(probabilities)
    metrics=classification_metrics(y,prob)
    metrics.update(unweighted_validation_ce=loss_summary(losses,y,None),
        sqrt_weighted_validation_ce=loss_summary(losses,y,original_weights),
        forward_seconds=forward,inference_ms_per_beat=forward*1000/len(labels),total_pass_seconds=time.perf_counter()-started,
        classes_with_nonzero_recall=sum(r['recall']>0 for r in metrics['per_class']),
        checkpoint_sha256=digest,parameters_unchanged=True,final_test_read=False)
    groups={g:classification_metrics(y[[r['patient_group']==g for r in metadata]],prob[[r['patient_group']==g for r in metadata]])
            for g in sorted({r['patient_group'] for r in metadata})}
    return metrics,groups,metadata,y,prob,np.asarray(losses)
