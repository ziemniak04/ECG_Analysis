"""Phase-24 Checkpoint A: metadata-only RR export and ONE real TRAIN smoke update."""
import argparse
from pathlib import Path
import json
import random
import time
import numpy as np
import torch
import wfdb
from torch.utils.data import DataLoader,Subset
from src.mitbih_rr_dataset import export_rr,MITBIHRRDataset,collate_rr,DEFAULT_RR_ROOT
from src.mitbih_rr_config import RRFeatureConfig,RR_FEATURE_NAMES
from src.mitbih_protocol import MITBIHExportIndex,load_split
from src.mitbih_training import loss_weights
from src.data_loader import DEFAULT_DATA_DIR
from src.visualization import BEAT_SYMBOLS
from src.mitbih_rr import rr_features,transform_rr
from src.models.mitbih_cnn_rr import MITBIHBeatCNNRR
from src.mitbih_full_run import snapshot
from src.prepare_dataset import sha256_file
from src.ptbxl_training import write_json_new


def run(output,rr_root):
    output,rr_root=Path(output),Path(rr_root)
    if output.exists() or rr_root.exists():raise FileExistsError('New setup run and versioned RR sidecar required')
    output.mkdir(parents=True,exist_ok=False);snapshot(output)
    metadata=export_rr(rr_root)
    index=MITBIHExportIndex();split=load_split(index)
    ds=MITBIHRRDataset(rr_root,partition='train',index=index,split=split)
    # Technical class coverage only: selection by y, never a feature input.
    wanted={i:[] for i in range(5)};groups={i:set() for i in range(5)};offset=0
    for entry in ds.base.shards:
        with np.load(index.root/entry.file,allow_pickle=False) as f:y=f['y']
        for c in range(5):
            found=np.flatnonzero(y==c)
            if len(wanted[c])<2 and len(found) and entry.patient_group not in groups[c]:
                wanted[c].append(offset+int(found[0]));groups[c].add(entry.patient_group)
        offset+=entry.count
    if any(len(v)!=2 for v in wanted.values()):raise ValueError('Training smoke class support insufficient')
    selected=sorted(i for values in wanted.values() for i in values)
    torch.manual_seed(42);random.seed(42);np.random.seed(42)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    batch=next(iter(DataLoader(Subset(ds,selected),batch_size=10,shuffle=False,num_workers=0,collate_fn=collate_rr)))
    assert tuple(batch['X'].shape)==(10,2,252) and tuple(batch['RR'].shape)==(10,6)
    assert sorted(batch['y'].tolist())==[0,0,1,1,2,2,3,3,4,4]
    assert set(batch['patient_group'])<=set(split['groups']['train'])
    # Recompute actual selected timing directly from original per-record ATR.
    for rid in sorted(set(batch['record_id'])):
        annotation=wfdb.rdann(str(DEFAULT_DATA_DIR/rid),'atr');header=wfdb.rdheader(str(DEFAULT_DATA_DIR/rid))
        positions=np.asarray(annotation.sample,dtype=np.int64)
        valid_positions=positions[np.array([s in BEAT_SYMBOLS for s in annotation.symbol])]
        selected_rows=[j for j,r in enumerate(batch['record_id']) if r==rid]
        raw,valid=rr_features(valid_positions,batch['annotation_position'][selected_rows].numpy(),header.fs)
        np.testing.assert_array_equal(raw,batch['rr_raw'][selected_rows].numpy())
        np.testing.assert_array_equal(valid,batch['rr_valid'][selected_rows].numpy())
    model=MITBIHBeatCNNRR();optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    weights=torch.from_numpy(loss_weights(index,split,'sqrt_balanced'))
    before={k:v.detach().clone() for k,v in model.named_parameters()}
    model.train();optimizer.zero_grad(set_to_none=True);start=time.perf_counter()
    logits=model(batch['X'],batch['RR']);loss=torch.nn.CrossEntropyLoss(weight=weights)(logits,batch['y'])
    if logits.shape!=(10,5) or not torch.isfinite(logits).all() or not torch.isfinite(loss):raise RuntimeError('Invalid smoke outputs')
    loss.backward()
    if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):raise RuntimeError('Invalid gradients')
    gradient_norm=float(torch.sqrt(sum((p.grad*p.grad).sum() for p in model.parameters())))
    if gradient_norm<=0:raise RuntimeError('Zero gradients')
    optimizer.step()
    if not all(torch.isfinite(p).all() for p in model.parameters()):raise RuntimeError('Nonfinite parameter update')
    if not any(not torch.equal(v.detach(),before[k]) for k,v in model.named_parameters()):raise RuntimeError('No parameter update')
    elapsed=time.perf_counter()-start
    model.eval()
    with torch.inference_mode():reference=model(batch['X'],batch['RR']).clone()
    checkpoint=output/'smoke_one_update_not_trained_model.pt'
    with checkpoint.open('xb') as f:torch.save(dict(schema='mitbih-cnn-rr-smoke-v1',model_state_dict=model.state_dict(),
        configuration=model.configuration(),optimizer_steps=1,not_for_performance_evaluation=True,
        rr_manifest_sha256=sha256_file(rr_root/'manifest.json'),class_names=list('NSVFQ')),f)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    restored=MITBIHBeatCNNRR();restored.load_state_dict(saved['model_state_dict']);restored.eval()
    with torch.inference_mode():torch.testing.assert_close(restored(batch['X'],batch['RR']),reference,rtol=0,atol=0)
    result=dict(status='passed',input_shape=list(batch['X'].shape),rr_shape=list(batch['RR'].shape),
        output_shape=list(logits.shape),parameter_count=model.parameter_count(),configuration=model.configuration(),
        RR_features=list(RR_FEATURE_NAMES),loss=float(loss.detach()),gradient_norm=gradient_norm,
        optimizer_steps=1,finite_gradients=True,parameters_changed=True,checkpoint_reload_identical_logits=True,
        checkpoint_sha256=sha256_file(checkpoint),seed=42,cpu_threads=2,device='cpu',seconds=elapsed,
        waveform_records_read=index.read_records,validation_forward_calls=0,test_forward_calls=0,
        source_beats=[dict(beat_id=batch['beat_id'][j],record_id=batch['record_id'][j],
            patient_group=batch['patient_group'][j],source_row=int(batch['source_row'][j]),
            annotation_position=int(batch['annotation_position'][j]),label=int(batch['y'][j]),
            rr_raw=batch['rr_raw'][j].tolist(),rr_valid=batch['rr_valid'][j].tolist(),RR=batch['RR'][j].tolist()) for j in range(10)],
        full_training_started=False,no_classification_performance_measured=True,
        rr_manifest_sha256=sha256_file(rr_root/'manifest.json'),rr_partition_summary=metadata['summary'])
    write_json_new(output/'smoke_result.json',result)
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--rr-root',type=Path,default=DEFAULT_RR_ROOT)
    args=p.parse_args();run(args.output_dir,args.rr_root)
