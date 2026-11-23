"""Phase-16 metadata audit and exactly one optional training-batch optimizer step.

No epochs, validation/test scoring, checkpoint export or ECG reprocessing.
If PyTorch is missing, save an explicit blocked result and exit with status 2.
"""
import argparse
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import numpy as np
from src.ptbxl_protocol import (PTBXLExportIndex,PARTITION_FOLDS,
                               training_class_counts,training_class_weights)
from src.ptbxl_model_config import RhythmCNNConfig


def write_new(path,data):
    with Path(path).open('x') as f:json.dump(data,f,indent=2,sort_keys=True,allow_nan=False)


def metadata_audit(index):
    counts=training_class_counts(index,index.partitions.train)
    weights=training_class_weights(index,index.partitions.train)
    groups={}
    for name in PARTITION_FOLDS:
        indices=getattr(index.partitions,name)
        groups[name]={'folds':list(PARTITION_FOLDS[name]),'record_count':len(indices),
                      'patient_count':len({index.entries[i].patient_id for i in indices}),
                      'indices':list(indices),'ecg_ids':[index.entries[i].ecg_id for i in indices]}
    # Audit actual waveforms from training only: one deterministic record per class.
    samples=[]
    for label in range(4):
        position=next(i for i in index.partitions.train if index.entries[i].label==label)
        arrays=index.load_numpy(position)
        entry=index.entries[position]
        samples.append({**asdict(entry),'shape':list(arrays['X'].shape),'X_dtype':str(arrays['X'].dtype),
                        'y_dtype':str(arrays['y'].dtype),'finite':bool(np.isfinite(arrays['X']).all())})
    return {'manifest':str(index.manifest_path.relative_to(Path.cwd())),
        'manifest_sha256':index.manifest_sha256,'record_count':len(index),
        'patient_count':len({e.patient_id for e in index.entries}),
        'class_names':['SR','AFIB','STACH','SBRAD'],'partitions':groups,
        'patient_overlap':False,'partition_rule':'Original folds 1-8 train, 9 validation, 10 test',
        'training_class_counts':counts.tolist(),'optional_training_balanced_weights':weights.tolist(),
        'weight_formula':'n_training/(4*n_training_class)','real_training_record_numpy_checks':samples}


def smoke_step(index,*,batch_size=4,seed=42,use_class_weights=False):
    """Exactly one CPU optimizer update on the first deterministic training batch."""
    import torch
    from torch.utils.data import DataLoader
    from src.ptbxl_torch import PTBXLRhythmDataset
    from src.models.ptbxl_cnn import PTBXLRhythmCNN
    if isinstance(batch_size,bool) or not isinstance(batch_size,int) or not 1<=batch_size<=8:
        raise ValueError('Smoke batch_size must be 1..8; no epoch loop is allowed here')
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    device=torch.device('cpu')
    dataset=PTBXLRhythmDataset(index,partition='train')
    loader=DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=0)
    batch=next(iter(loader))
    if not all(int(fold) in PARTITION_FOLDS['train'] for fold in batch['strat_fold']):
        raise RuntimeError('Smoke batch contains a non-training record')
    x,y=batch['X'].to(device),batch['y'].to(device)
    assert tuple(x.shape)==(batch_size,12,1000) and y.dtype==torch.long
    assert bool(((y>=0)&(y<4)).all())
    model=PTBXLRhythmCNN().to(device)
    model.train()
    weights=torch.from_numpy(training_class_weights(index,index.partitions.train)).to(device) if use_class_weights else None
    criterion=torch.nn.CrossEntropyLoss(weight=weights)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    before=[parameter.detach().clone() for parameter in model.parameters()]
    optimizer.zero_grad(set_to_none=True)
    logits=model(x)
    assert tuple(logits.shape)==(batch_size,4) and bool(torch.isfinite(logits).all())
    loss=criterion(logits,y)
    assert bool(torch.isfinite(loss))
    loss.backward()
    gradients=[p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None and bool(torch.isfinite(g).all()) for g in gradients)
    assert sum(float(g.detach().norm()) for g in gradients)>0
    optimizer.step()
    changed=any(not torch.equal(old,new.detach()) for old,new in zip(before,model.parameters()))
    assert changed
    return {'status':'passed','torch_version':torch.__version__,'device':str(device),
        'cuda_available_observed':torch.cuda.is_available(),'seed':seed,'optimizer_updates':1,
        'batch_ecg_ids':batch['ecg_id'].tolist(),'batch_patient_ids':batch['patient_id'].tolist(),
        'batch_strat_folds':batch['strat_fold'].tolist(),'input_shape':list(x.shape),
        'logits_shape':list(logits.shape),'parameter_count':model.parameter_count(),
        'finite_logits':True,'finite_loss':True,'finite_nonzero_gradients':True,
        'parameters_changed':changed,'smoke_loss':float(loss.detach()),'class_weights_enabled':use_class_weights,
        'note':'Technical initialization check only; no accuracy or generalization claim, no checkpoint saved.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('results/model_setup/phase16'))
    parser.add_argument('--audit-only',action='store_true')
    parser.add_argument('--class-weights',action='store_true')
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    index=PTBXLExportIndex()
    audit=metadata_audit(index)
    write_new(args.output_dir/'partition_protocol.json',audit)
    configuration=RhythmCNNConfig()
    environment={'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
                 'packages':{name:importlib.metadata.version(name) for name in ('numpy','wfdb','scipy','pandas','pytest')}}
    if args.audit_only:
        result={'status':'audit_only','smoke_executed':False}
    else:
        try:
            result=smoke_step(index,use_class_weights=args.class_weights)
            result['smoke_executed']=True
        except ModuleNotFoundError as exc:
            if exc.name!='torch':raise
            result={'status':'blocked_missing_pytorch','smoke_executed':False,'optimizer_updates':0,
                'reason':'PyTorch is unavailable; CPU installation was blocked in this environment.',
                'manual_install_command':'.venv/bin/python -m pip install -r requirements_model_cpu.txt'}
    write_new(args.output_dir/'smoke_result.json',{**result,'environment':environment,
        'model_configuration':configuration.to_dict(),'configured_parameter_count':configuration.expected_parameter_count()})
    print(json.dumps({k:audit[k] for k in ('record_count','patient_count','training_class_counts','optional_training_balanced_weights')},indent=2))
    print(json.dumps(result,indent=2))
    if result['status']=='blocked_missing_pytorch':raise SystemExit(2)


if __name__=='__main__':
    main()
