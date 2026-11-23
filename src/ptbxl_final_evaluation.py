"""One authorized final evaluation of the frozen Phase-18C B checkpoint.

No training, preprocessing, threshold parameters or alternative checkpoints.
Claim before test forward; keep batch predictions so interrupted work is auditable.
Never automatically replay a partially evaluated test cohort.
"""
from datetime import datetime, timezone
import json
from numbers import Integral
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from .models.ptbxl_cnn import PTBXLRhythmCNN
from .ptbxl_model_config import RhythmCNNConfig
from .ptbxl_protocol import PTBXLExportIndex, file_sha256
from .ptbxl_torch import PTBXLRhythmDataset
from .ptbxl_metrics import CLASS_NAMES
from .ptbxl_training import write_json_new
from .ptbxl_full_run import atomic_status

CHECKPOINT = Path('results/model_training/phase18c_optimization/run_001/B_unweighted/best_model.pt')
CHECKPOINT_SHA256 = '40dab87717a6c563e34f9c77e546b5f6f35f44eab73b4357bd4ed0181fc1351e'
PHASE19_ROOT = Path('results/model_evaluation/phase19_final_test')
ONCE_CLAIM = PHASE19_ROOT/'evaluation_claim.json'
DEFAULT_OUTPUT = PHASE19_ROOT/'run_001'


def claim_once(path, identity):
    """Exclusive durable claim, retained even on failure; never overwrite/unlink."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as target:
        json.dump(identity,target,indent=2,allow_nan=False)
        target.flush();os.fsync(target.fileno())


class TestOnlyIndex(PTBXLExportIndex):
    """Read only official fold 10; reuse all existing checksum/metadata validation."""
    __test__ = False
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.read_ecg_ids=[];self.blocked_reads=0
    def load_numpy(self,position,*,verify_checksum=True):
        if (isinstance(position,bool) or not isinstance(position,Integral)
                or not 0<=position<len(self.entries)):
            raise IndexError('Invalid source index')
        if self.entries[position].strat_fold!=10:
            self.blocked_reads+=1
            raise RuntimeError('Final evaluator permits only official test fold 10')
        value=super().load_numpy(position,verify_checksum=verify_checksum)
        self.read_ecg_ids.append(self.entries[position].ecg_id)
        return value


def infer_test_batches(model,index,*,batch_size=16,batch_sink=None):
    """Exactly one ordered pass, no warmup/reload forward on any test recording.

    The caller must acquire the once-only claim before invoking this function
    on real data. Unit tests use explicit synthetic export fixtures only.
    """
    if not isinstance(index,TestOnlyIndex):raise TypeError('Test-only guarded index required')
    if isinstance(batch_size,bool) or not isinstance(batch_size,int) or batch_size<1:
        raise ValueError('Positive batch size required')
    before={key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
    model.eval()
    for parameter in model.parameters():parameter.requires_grad_(False)
    loader=DataLoader(PTBXLRhythmDataset(index,partition='test'),batch_size=batch_size,
                      shuffle=False,num_workers=0,drop_last=False)
    chunks=[];times=[];wall_start=time.perf_counter()
    with torch.inference_mode():
        for number,batch in enumerate(loader,1):
            if not bool((batch['strat_fold']==10).all()):raise RuntimeError('Wrong official fold in test batch')
            start=time.perf_counter();logits=model(batch['X']);forward=time.perf_counter()-start
            if tuple(logits.shape)!=(len(batch['y']),4) or not bool(torch.isfinite(logits).all()):
                raise ValueError('Invalid frozen-model logits')
            probability=torch.softmax(logits,dim=1).cpu().numpy().astype(np.float64)
            chunk={'labels':batch['y'].cpu().numpy(),'probabilities':probability,
                   'predicted_labels':probability.argmax(axis=1).astype(np.int64),
                   'ecg_id':batch['ecg_id'].numpy(),'patient_id':batch['patient_id'].numpy(),
                   'strat_fold':batch['strat_fold'].numpy()}
            timing={'batch':number,'records':len(batch['y']),'forward_seconds':forward}
            if batch_sink is not None:batch_sink(number,chunk,timing)
            chunks.append(chunk);times.append(timing)
            if number==1 or number%20==0 or number==len(loader):
                print(f'FINAL TEST: batch {number}/{len(loader)}, records {sum(t["records"] for t in times)}',flush=True)
    wall=time.perf_counter()-wall_start
    if not chunks:raise ValueError('Empty test cohort')
    arrays={key:np.concatenate([chunk[key] for chunk in chunks]) for key in chunks[0]}
    expected=[index.entries[i].ecg_id for i in index.partitions.test]
    if arrays['ecg_id'].tolist()!=expected or len(set(expected))!=len(expected):
        raise RuntimeError('Test coverage/order/identity mismatch')
    if any(not torch.equal(value.detach().cpu(),before[key]) for key,value in model.state_dict().items()):
        raise RuntimeError('Frozen model parameters or buffers changed')
    if any(p.grad is not None for p in model.parameters()):raise RuntimeError('Unexpected gradients')
    timing={'batches':times,'forward_total_seconds':sum(t['forward_seconds'] for t in times),
            'inference_wall_seconds_including_loading_and_batch_persistence':wall,
            'forward_ms_per_record':1000*sum(t['forward_seconds'] for t in times)/len(expected),
            'model_parameters_and_buffers_unchanged':True,'gradients_created':False,
            'warmup_test_forwards':0,'waveform_passes':1,'coverage':1.,'abstentions':0}
    return arrays,timing


def prepare(output=DEFAULT_OUTPUT):
    """Metadata-only preflight; no test waveform or model forward is read/run."""
    output=Path(output)
    if output.exists():raise FileExistsError('A NEW Phase-19 output directory is required')
    if ONCE_CLAIM.exists():raise FileExistsError('Final test evaluation already claimed; never repeat inference')
    if file_sha256(CHECKPOINT)!=CHECKPOINT_SHA256:raise ValueError('Approved checkpoint SHA256 differs')
    run=CHECKPOINT.parent
    selected=json.loads((run.parent/'frozen_candidate.json').read_text())
    if (Path(selected['checkpoint'])!=CHECKPOINT or selected['checkpoint_sha256']!=CHECKPOINT_SHA256
            or selected['sbrad_threshold']!=0 or selected['class_names']!=list(CLASS_NAMES)):
        raise ValueError('Frozen Phase-18C selection differs from authorized argmax B')
    training=json.loads((run/'selection_manifest.json').read_text())
    config=json.loads((run/'training_config.json').read_text())
    if config['class_weighted_loss'] or training['class_weights'] is not None:
        raise ValueError('Selected model must be unweighted B')
    for name,sha in training['code_sha256'].items():
        if file_sha256(Path('src')/name)!=sha:raise ValueError(f'Training code changed: {name}')
    index=TestOnlyIndex()
    if index.manifest_sha256!=training['manifest_sha256']:raise ValueError('Dataset manifest changed')
    for partition in ('train','validation'):
        ids=[index.entries[i].ecg_id for i in getattr(index.partitions,partition)]
        if ids!=training['partitions'][partition]['ecg_ids']:raise ValueError('Original populations differ')
    test=[index.entries[i] for i in index.partitions.test]
    rest=[entry for entry in index.entries if entry.strat_fold!=10]
    patients={entry.patient_id for entry in test}
    if len(test)!=1958 or len(patients)!=1711:raise ValueError('Original test cohort count differs')
    if patients & {entry.patient_id for entry in rest}:raise ValueError('Patient leakage into test cohort')
    payload=torch.load(CHECKPOINT,map_location='cpu',weights_only=True)
    expected=json.loads(json.dumps(RhythmCNNConfig().to_dict()))
    if (payload['schema']!='ptbxl-best-model-v1' or payload['classes']!=list(CLASS_NAMES)
            or json.loads(json.dumps(payload['model_configuration']))!=expected
            or payload['manifest_sha256']!=index.manifest_sha256):
        raise ValueError('Checkpoint architecture, classes or source differs')
    model=PTBXLRhythmCNN();model.load_state_dict(payload['model_state_dict'],strict=True)
    if model.parameter_count()!=74116:raise ValueError('Parameter count differs')
    identity=json.loads((index.root/'dataset_identity.json').read_text())
    for name,sha in identity['processing_code_sha256'].items():
        if file_sha256(Path('src')/name)!=sha:raise ValueError(f'Export preprocessing code changed: {name}')
    output.mkdir(parents=True,exist_ok=False)
    protected=[]
    for folder in ('src','tests','examples','configs'):
        protected.extend(p for p in Path(folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    protected.extend(p for p in Path('.').iterdir() if p.is_file())
    for folder in ('results/model_training','results/model_evaluation/phase18b_validation'):
        protected.extend(p for p in Path(folder).rglob('*') if p.is_file())
    write_json_new(output/'preservation_before.json',{str(p):file_sha256(p) for p in protected})
    stats={str(p):[p.stat().st_size,p.stat().st_mtime_ns]
           for folder in ('data/processed/ecg_dataset_v1','data/processed/ptbxl_rhythm_v1')
           for p in Path(folder).rglob('*') if p.is_file()}
    write_json_new(output/'dataset_stats_before.json',stats)
    population=[{'ecg_id':e.ecg_id,'patient_id':e.patient_id,'strat_fold':e.strat_fold,
                 'label':e.label,'class_name':e.rhythm_code,'file':e.file,'sha256':e.sha256} for e in test]
    write_json_new(output/'test_cohort_manifest.json',population)
    preflight={'status':'passed','checkpoint':str(CHECKPOINT),'checkpoint_sha256':CHECKPOINT_SHA256,
        'architecture':expected,'parameters':74116,'input_shape':[12,1000],
        'input_dtype':'float32','class_names':list(CLASS_NAMES),'decision_rule':'argmax',
        'thresholds_applied':False,'abstention_enabled':False,'batch_size':16,'cpu_threads':2,
        'device':'cpu','seed':42,'manifest':str(index.manifest_path),'manifest_sha256':index.manifest_sha256,
        'preprocessing_configuration':index.configuration,'test_records':1958,'test_patients':1711,
        'train_patients':len({index.entries[i].patient_id for i in index.partitions.train}),
        'validation_patients':len({index.entries[i].patient_id for i in index.partitions.validation}),
        'patient_overlap_with_folds_1_to_9':False,'waveforms_read_during_preflight':0,
        'evaluator_sha256':file_sha256(Path(__file__)),
        'training_code_sha256':training['code_sha256'],
        'cohort_manifest_sha256':file_sha256(output/'test_cohort_manifest.json'),
        'created_utc':datetime.now(timezone.utc).isoformat()}
    write_json_new(output/'preflight.json',preflight)
    return preflight


def evaluate_once(output=DEFAULT_OUTPUT):
    output=Path(output)
    if ONCE_CLAIM.exists():raise FileExistsError('Final evaluation already claimed; use saved predictions only')
    pre=json.loads((output/'preflight.json').read_text())
    if pre['status']!='passed' or file_sha256(CHECKPOINT)!=CHECKPOINT_SHA256:
        raise ValueError('Preflight/checkpoint differs')
    if file_sha256(Path(__file__))!=pre['evaluator_sha256']:
        raise ValueError('Evaluator code changed after preflight')
    for name,sha in pre['training_code_sha256'].items():
        if file_sha256(Path('src')/name)!=sha:raise ValueError('Frozen training/model code changed')
    if file_sha256(output/'test_cohort_manifest.json')!=pre['cohort_manifest_sha256']:
        raise ValueError('Approved test cohort changed')
    index=TestOnlyIndex()
    if index.manifest_sha256!=pre['manifest_sha256'] or index.configuration!=pre['preprocessing_configuration']:
        raise ValueError('Export differs from preflight')
    expected=json.loads((output/'test_cohort_manifest.json').read_text())
    if [(e.ecg_id,e.patient_id,e.label) for e in (index.entries[i] for i in index.partitions.test)] != [
            (e['ecg_id'],e['patient_id'],e['label']) for e in expected]:raise ValueError('Test identity differs')
    torch.manual_seed(42);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    model=PTBXLRhythmCNN()
    model.load_state_dict(torch.load(CHECKPOINT,map_location='cpu',weights_only=True)['model_state_dict'],strict=True)
    claim_once(ONCE_CLAIM,{'output':str(output),'checkpoint_sha256':CHECKPOINT_SHA256,
        'cohort_manifest_sha256':pre['cohort_manifest_sha256'],'decision_rule':'argmax',
        'created_utc':datetime.now(timezone.utc).isoformat(),'authorization':'User Phase-19 one final evaluation'})
    (output/'batches').mkdir(exist_ok=False)
    def save_batch(number,chunk,timing):
        with (output/'batches'/f'batch_{number:03d}.npz').open('xb') as target:
            np.savez_compressed(target,**chunk);target.flush();os.fsync(target.fileno())
        atomic_status(output,status='inference_running',completed_batch=number,
                      completed_records=len(index.read_ecg_ids))
    try:
        arrays,timing=infer_test_batches(model,index,batch_size=16,batch_sink=save_batch)
        if len(arrays['labels'])!=1958 or len(index.read_ecg_ids)!=1958 or index.blocked_reads:
            raise RuntimeError('Final test did not cover exactly the approved cohort')
        with (output/'test_predictions.npz').open('xb') as target:
            np.savez_compressed(target,**arrays);target.flush();os.fsync(target.fileno())
        if file_sha256(CHECKPOINT)!=CHECKPOINT_SHA256:raise RuntimeError('Checkpoint file changed')
        timing.update({'checkpoint_sha256_before_and_after':CHECKPOINT_SHA256,
            'waveform_folds_read':[10],'read_records':1958,'unique_read_records':len(set(index.read_ecg_ids)),
            'blocked_reads':index.blocked_reads,'torch':str(torch.__version__),'numpy':np.__version__,
            'completed_utc':datetime.now(timezone.utc).isoformat()})
        write_json_new(output/'inference_integrity.json',timing)
        atomic_status(output,status='inference_completed',records=1958,model_unchanged=True)
        return timing
    except BaseException as exc:
        atomic_status(output,status='inference_failed_do_not_automatically_repeat',
            error=f'{type(exc).__name__}: {exc}',records_read=len(index.read_ecg_ids))
        raise
