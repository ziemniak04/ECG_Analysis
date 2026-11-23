"""Phase 20: metadata audit, freeze patient split, one TRAINING optimizer update."""
import argparse
import csv
import json
from pathlib import Path
import platform
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from src.mitbih_protocol import (MITBIHExportIndex,inventory,propose_split,load_split,
    validate_split,training_class_weights,DEFAULT_SPLIT)
from src.mitbih_torch import MITBIHBeatDataset,collate_mitbih
from src.mitbih_model_config import BeatCNNConfig
from src.models.mitbih_cnn import MITBIHBeatCNN
from src.prepare_dataset import sha256_file,write_json
from src.label_mapping import CLASS_NAMES,DEFAULT_MAPPING

DEFAULT_OUTPUT=Path('results/model_setup/phase20_mitbih/run_002')


def csv_new(path,rows):
    with path.open('x',newline='') as target:
        writer=csv.DictWriter(target,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def audit(output=DEFAULT_OUTPUT):
    output=Path(output)
    if output.exists():raise FileExistsError('Use a new Phase-20 directory')
    index=MITBIHExportIndex();output.mkdir(parents=True,exist_ok=False)
    # Freeze completed work, not the new Phase-20 implementation being developed.
    protected=[p for p in Path('.').iterdir() if p.is_file()]
    for folder in ('src','tests','examples'):
        protected.extend(p for p in Path(folder).rglob('*.py')
            if 'mitbih' not in p.name and '__pycache__' not in p.parts)
    for folder in ('results/model_training','results/model_evaluation'):
        protected.extend(p for p in Path(folder).rglob('*') if p.is_file())
    write_json(output/'preservation_before.json',{str(p):sha256_file(p) for p in protected})
    stats={str(p):[p.stat().st_size,p.stat().st_mtime_ns]
        for folder in ('data/processed/ecg_dataset_v1','data/processed/ptbxl_rhythm_v1')
        for p in Path(folder).rglob('*') if p.is_file()}
    write_json(output/'dataset_stats_before.json',stats)
    actual={}
    for e in index.entries:
        path=index.root/e.file
        if sha256_file(path)!=e.sha256:raise ValueError('Export SHA256 mismatch')
        # Label audit before split design: never decode the waveform-bearing X.
        with np.load(path,allow_pickle=False) as f:
            y=f['y'];symbols=f['original_symbols'];groups=f['patient_groups'];records=f['record_ids']
            if (y.dtype!=np.int64 or len(y)!=e.count or not np.isin(y,range(5)).all()
                    or np.bincount(y,minlength=5).tolist()!=list(e.class_counts)
                    or not np.all(groups==e.patient_group) or not np.all(records==e.record_id)
                    or [DEFAULT_MAPPING.get(s) for s in symbols]!=[CLASS_NAMES[int(i)] for i in y]):
                raise ValueError(f'Label/metadata mismatch in {e.record_id}')
        actual[e.record_id]=list(e.class_counts)
    info=inventory(index);proposal=propose_split(index)
    write_json(output/'patient_class_audit.json',info);write_json(output/'proposed_split.json',proposal)
    csv_new(output/'patient_class_distribution.csv',[
        {'patient_group':r['patient_group'],'record_ids':','.join(r['record_ids']),'beats':r['beats'],**r['counts']}
        for r in info['per_group']])
    csv_new(output/'record_class_distribution.csv',[
        {'record_id':e.record_id,'patient_group':e.patient_group,'beats':e.count,
         **dict(zip(CLASS_NAMES,e.class_counts))} for e in index.entries])
    write_json(output/'audit_result.json',{'status':'passed','records':48,'patient_groups':47,'beats':109452,
        'metadata_counts_verified_against_all_48_npz_labels':True,'waveform_arrays_read':0,
        'F_two_largest_group_beats':sum(r[1] for r in info['per_class'][3]['groups_descending'][:2]),
        'Q_four_largest_group_beats':sum(r[1] for r in info['per_class'][4]['groups_descending'][:4]),
        'manifest_sha256':index.manifest_sha256,'proposal_sha256':sha256_file(output/'proposed_split.json')})
    print(json.dumps({'audit':'passed','proposed_split':proposal},indent=2),flush=True)
    return proposal


def freeze(output=DEFAULT_OUTPUT):
    output=Path(output);index=MITBIHExportIndex()
    result=json.loads((output/'audit_result.json').read_text())
    if result['status']!='passed' or sha256_file(output/'proposed_split.json')!=result['proposal_sha256']:
        raise ValueError('Audit/proposal changed')
    split=json.loads((output/'proposed_split.json').read_text());validate_split(index,split)
    write_json(DEFAULT_SPLIT,split)  # Exclusive creation, no later model-based changes.
    write_json(output/'frozen_split_identity.json',{'file':str(DEFAULT_SPLIT),'sha256':sha256_file(DEFAULT_SPLIT),
        'test_waveforms_used_for_model_setup':0,'selection_basis':'labels and verified groups only'})
    csv_new(output/'partition_class_support.csv',[
        {'partition':p,'groups':r['group_count'],'recordings':r['record_count'],'beats':r['beats'],
         **r['class_counts'],**{c+'_groups':r['supporting_groups'][c] for c in CLASS_NAMES}}
        for p,r in split['summary'].items()])
    return split


def smoke(output=DEFAULT_OUTPUT):
    output=Path(output)
    if (output/'smoke_result.json').exists() or (output/'smoke_after_one_update.pt').exists():
        raise FileExistsError('Technical smoke already executed; never overwrite')
    index=MITBIHExportIndex();split=load_split(index)
    identity=json.loads((output/'frozen_split_identity.json').read_text())
    if sha256_file(DEFAULT_SPLIT)!=identity['sha256']:raise ValueError('Frozen split changed')
    dataset=MITBIHBeatDataset(index,split,partition='train',max_cached_shards=2)
    # Two actual training beats per class from two groups when available.
    selected=[];offset=0;needed={i:[] for i in range(5)};used={i:set() for i in range(5)}
    for e in dataset.shards:
        with np.load(index.root/e.file,allow_pickle=False) as f:y=f['y']
        for c in range(5):
            matches=np.flatnonzero(y==c)
            if len(needed[c])<2 and len(matches) and e.patient_group not in used[c]:
                needed[c].append(offset+int(matches[0]));used[c].add(e.patient_group)
        offset+=e.count
    if any(len(v)!=2 for v in needed.values()):raise ValueError('Insufficient class/group support for smoke')
    selected=sorted(i for values in needed.values() for i in values)
    torch.manual_seed(42);np.random.seed(42);random.seed(42)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    batch=next(iter(DataLoader(Subset(dataset,selected),batch_size=10,shuffle=False,
                              num_workers=0,collate_fn=collate_mitbih)))
    assert tuple(batch['X'].shape)==(10,2,252) and batch['X'].dtype==torch.float32
    assert batch['y'].dtype==torch.long and sorted(batch['y'].tolist())==[0,0,1,1,2,2,3,3,4,4]
    assert set(batch['patient_group'])<=set(split['groups']['train'])
    weights=training_class_weights(index,split,split['groups']['train'])
    model=MITBIHBeatCNN();optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    before={k:v.detach().clone() for k,v in model.named_parameters()}
    model.train();optimizer.zero_grad(set_to_none=True);start=time.perf_counter()
    logits=model(batch['X'])
    loss=torch.nn.CrossEntropyLoss(weight=torch.tensor(weights))(logits,batch['y'])
    if tuple(logits.shape)!=(10,5) or not torch.isfinite(logits).all() or not torch.isfinite(loss):
        raise RuntimeError('Invalid smoke outputs/loss')
    loss.backward();grads=[p.grad for p in model.parameters()]
    if not all(g is not None and torch.isfinite(g).all() for g in grads):raise RuntimeError('Invalid gradients')
    grad_norm=float(torch.sqrt(sum(torch.sum(g*g) for g in grads)))
    if grad_norm<=0:raise RuntimeError('All gradients zero')
    optimizer.step()
    if not all(torch.isfinite(p).all() for p in model.parameters()):raise RuntimeError('Nonfinite updated parameters')
    changed=any(not torch.equal(v.detach(),before[k]) for k,v in model.named_parameters())
    if not changed:raise RuntimeError('Optimizer did not change parameters')
    elapsed=time.perf_counter()-start
    path=output/'smoke_after_one_update.pt'
    with path.open('xb') as target:torch.save({'schema':'mitbih-technical-smoke-not-trained-model-v1',
        'model_configuration':BeatCNNConfig().to_dict(),'model_state_dict':model.state_dict(),
        'class_names':list(CLASS_NAMES),'split_sha256':identity['sha256'],
        'optimizer_steps':1,'not_for_evaluation_or_application':True},target)
    restored=torch.load(path,map_location='cpu',weights_only=True)
    if not all(torch.equal(v,restored['model_state_dict'][k]) for k,v in model.state_dict().items()):
        raise RuntimeError('Checkpoint tensor roundtrip failed')
    selected_meta=[{'record_id':batch['record_id'][j],'patient_group':batch['patient_group'][j],
        'symbol':batch['original_symbol'][j],'annotation_position':int(batch['annotation_position'][j]),
        'annotation_index':int(batch['annotation_index'][j]),'class_index':int(batch['y'][j]),
        'source_row':int(batch['source_row'][j])} for j in range(10)]
    result={'status':'passed','model_configuration':BeatCNNConfig().to_dict(),
        'parameter_count':model.parameter_count(),'input_shape':list(batch['X'].shape),
        'output_shape':list(logits.shape),'input_dtype':str(batch['X'].dtype),'label_dtype':str(batch['y'].dtype),
        'class_names':list(CLASS_NAMES),'training_only_class_weights':weights.tolist(),
        'class_weighting_enabled_for_technical_check':True,'loss':float(loss.detach()),
        'gradient_norm':grad_norm,'finite_nonzero_gradients':True,'parameters_changed':changed,
        'optimizer_steps':1,'smoke_seconds':elapsed,'seed':42,'cpu_threads':2,'device':'cpu',
        'sample_metadata':selected_meta,'waveform_records_read':index.read_records,
        'test_model_forward_calls':0,'validation_model_forward_calls':0,
        'split_sha256':identity['sha256'],'checkpoint_sha256':sha256_file(path),
        'checkpoint_roundtrip_equal':True,'torch':str(torch.__version__),'python':platform.python_version(),
        'not_a_training_experiment':True,'no_performance_metrics_computed':True}
    write_json(output/'smoke_result.json',result);write_json(output/'model_configuration.json',BeatCNNConfig().to_dict())
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('audit','freeze','smoke'))
    parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args()
    result={'audit':audit,'freeze':freeze,'smoke':smoke}[args.stage](args.output_dir)
    if args.stage!='audit':print(json.dumps(result,indent=2),flush=True)
