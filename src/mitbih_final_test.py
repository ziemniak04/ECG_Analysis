"""Single-authorized final MIT-BIH evaluation; no training or decision tuning.

A global exclusive claim precedes all held-out I/O. Failure consumes the claim:
never automatically rerun inference. Reporting uses saved predictions only.
"""
from pathlib import Path
import hashlib,json,os,sys,time,platform,importlib.metadata
import numpy as np
import torch
from .mitbih_protocol import MITBIHExportIndex,load_split,DEFAULT_SPLIT
from .mitbih_readiness import verify_frozen_protocol,resources
from .mitbih_rr_dataset import DEFAULT_RR_ROOT
from .mitbih_rr import rr_features,transform_rr,beat_ids
from .mitbih_rr_config import RR_FEATURE_NAMES,RRFeatureConfig
from .models.mitbih_cnn_rr import MITBIHBeatCNNRR
from .prepare_dataset import sha256_file
from .mitbih_training import fingerprint
from .mitbih_full_run import snapshot
from .ptbxl_training import write_json_new
from .data_loader import DEFAULT_DATA_DIR
from .visualization import BEAT_SYMBOLS
from .mitbih_execution_guard import HoldoutAccessGuard,require_guard

ROOT=Path('results/model_evaluation/phase26_mitbih_final_test')
CHECKPOINT=Path('results/model_training/phase24b_mitbih_cnn_rr/run_001/best_model.pt')
EXPECTED_SHA='9440f3494af59d3272d7c8de115c03b592d2738279957d97b5e7a6ac9c235536'


def claim_once(path,payload):
    """Create-before-access, never overwrite; failed attempts also consume access."""
    with Path(path).open('x') as f:
        json.dump(payload,f,indent=2);f.flush();os.fsync(f.fileno())
    Path(path).chmod(0o444)


def ecg_contract(config):
    """Record-specific ECG statistics/leads are expected to differ, method is fixed."""
    result=dict(config);n=dict(result.pop('normalization'))
    for key in ('offset','scale','source_ids','channel_names','sample_count'):n.pop(key)
    result['normalization_method_contract']=n
    return result


class SinglePassEngine:
    """Each record and beat processed once; labels are never model inputs."""
    def __init__(self,model,expected_counts,batch_size=32):
        self.model=model.eval();self.expected=dict(expected_counts);self.seen=set();self.batch_size=batch_size
        self.forward_seconds=0.;self.batch_count=0;self.beats=0
        self.before={k:v.clone() for k,v in model.state_dict().items()}
    def record(self,rid,x,rr):
        if rid not in self.expected or rid in self.seen:raise PermissionError('Unknown/repeated inference record')
        n=self.expected[rid]
        if x.shape!=(n,2,252) or rr.shape!=(n,6) or x.dtype!=np.float32 or rr.dtype!=np.float32 or not np.isfinite(x).all() or not np.isfinite(rr).all():raise ValueError('Invalid frozen model inputs')
        self.seen.add(rid)  # Mark before first forward, never retry a partially processed record.
        probabilities=[]
        with torch.inference_mode():
            for start in range(0,n,self.batch_size):
                X=torch.from_numpy(x[start:start+self.batch_size]);R=torch.from_numpy(rr[start:start+self.batch_size])
                tick=time.perf_counter();logits=self.model(X,R);self.forward_seconds+=time.perf_counter()-tick
                if logits.shape!=(len(X),5) or not torch.isfinite(logits).all():raise ValueError('Invalid logits')
                probabilities.append(torch.softmax(logits,1).numpy().copy());self.batch_count+=1;self.beats+=len(X)
        return np.concatenate(probabilities)
    def finish(self):
        if self.seen!=set(self.expected) or self.beats!=sum(self.expected.values()):raise ValueError('Incomplete test pass')
        if any(not torch.equal(self.before[k],v) for k,v in self.model.state_dict().items()):raise ValueError('Model parameters/buffers changed')
        return dict(beats=self.beats,record_count=len(self.seen),batch_count=self.batch_count,forward_seconds=self.forward_seconds,inference_ms_per_beat=1000*self.forward_seconds/self.beats,parameters_unchanged=True,scope='CPU forward only; excludes loading, RR prep, softmax and metrics')


def preflight(out):
    """Must execute under the original deny-all test guard. Metadata only."""
    require_guard()
    if (ROOT/'evaluation_claim.json').exists():raise PermissionError('Final evaluation already claimed; inference cannot repeat')
    if out.exists():raise FileExistsError('New Phase26 directory required')
    if sha256_file(CHECKPOINT)!=EXPECTED_SHA or CHECKPOINT.with_suffix('.sha256').read_text().strip()!=EXPECTED_SHA:raise ValueError('Checkpoint identity differs')
    saved=torch.load(CHECKPOINT,map_location='cpu',weights_only=True)
    plan=json.loads((CHECKPOINT.parent/'training_plan.json').read_text())
    selection=json.loads(Path('results/model_training/phase25b_reduced_f/run_001/FINAL_INTEGRITY.json').read_text())
    if selection['preferred_research_candidate']!='Phase24B' or selection['reference_sha256']!=EXPECTED_SHA:
        raise ValueError('Previously frozen model selection differs')
    model=MITBIHBeatCNNRR()
    if saved['schema']!='mitbih-cnn-rr-best-model-v1' or saved['model_configuration']!=model.configuration() or saved['class_names']!=list('NSVFQ') or saved['plan_sha256']!=fingerprint(plan):raise ValueError('Frozen model/plan differs')
    model.load_state_dict(saved['model_state_dict']);model.eval()
    if not all(torch.isfinite(t).all() for t in model.state_dict().values()):raise ValueError('Nonfinite checkpoint')
    index=MITBIHExportIndex();split=load_split(index);summary=verify_frozen_protocol(index,split)
    if summary['test']['beats']!=23392 or summary['test']['group_count']!=9 or [summary['test']['class_counts'][c] for c in 'NSVFQ']!=[19246,575,1402,363,1806]:raise ValueError('Test protocol differs')
    if index.manifest_sha256!=plan['manifest_sha256'] or split!=plan['split']:raise ValueError('Source/split differs')
    for name,digest in plan['code_sha256'].items():
        if sha256_file(Path('src')/name)!=digest:raise ValueError('Frozen source differs: '+name)
    manifest=json.loads((DEFAULT_RR_ROOT/'manifest.json').read_text())
    if sha256_file(DEFAULT_RR_ROOT/'manifest.json')!=saved['rr_manifest_sha256'] or saved['rr_manifest_sha256']!=plan['rr_manifest_sha256']:raise ValueError('RR manifest differs')
    for name,digest in manifest['artifact_sha256'].items():
        if sha256_file(DEFAULT_RR_ROOT/name)!=digest:raise ValueError('RR config/scaler checksum differs')
    state=json.loads((DEFAULT_RR_ROOT/'training_transform.json').read_text());rrconfig=json.loads((DEFAULT_RR_ROOT/'rr_configuration.json').read_text())
    if saved['preprocessing_configuration']!=dict(ECG=plan['ECG_preprocessing_by_record'],RR=rrconfig,RR_transform=state):raise ValueError('Preprocessing contract differs')
    if state!=plan['rr_training_transform'] or state['fitted_partition']!='train' or set(state['training_groups'])!=set(split['groups']['train']) or state['training_rows']!=62926:raise ValueError('Scaler fit identity differs')
    if rrconfig['model_feature_names']!=list(RR_FEATURE_NAMES) or rrconfig['context_policy']!='past_only; no next RR' or rrconfig['history_intervals']!=8 or rrconfig['valid_heartbeat_symbols']!=sorted(BEAT_SYMBOLS):raise ValueError('RR feature definition differs')
    contracts=[ecg_contract(c) for c in plan['ECG_preprocessing_by_record'].values()]
    if any(c!=contracts[0] for c in contracts):raise ValueError('Inconsistent frozen ECG methods')
    capacity=resources()
    if capacity['MemAvailable_bytes']<2*2**30 or capacity['disk_free_bytes']<128*2**20:raise RuntimeError('Insufficient resources')
    protocol=dict(schema='phase26-final-test-v1',checkpoint=str(CHECKPOINT),checkpoint_sha256=EXPECTED_SHA,model_configuration=saved['model_configuration'],class_names=list('NSVFQ'),source_manifest_sha256=index.manifest_sha256,frozen_split_sha256=sha256_file(DEFAULT_SPLIT),split=split,summary=summary,RR_configuration=rrconfig,RR_transform=state,RR_manifest_sha256=saved['rr_manifest_sha256'],ECG_method_contract=contracts[0],decision_rule='five-class argmax; no thresholds; no abstention',device='cpu',cpu_threads=2,batch_size=32,seed=42,dependencies={d.metadata['Name']:d.version for d in importlib.metadata.distributions() if d.metadata['Name']},python=sys.version,platform=platform.platform(),resources=capacity,authorization='User explicitly authorizes ONE Phase24B final held-out evaluation; feature preparation part of same attempt',historical_24A_incident='Nine test shards unintentionally read by legacy QC, without training, fitting, model inference or selection. Not a never-read test claim.',source_files_sha256={str(p):sha256_file(p) for p in [Path(__file__),Path('examples/evaluate_mitbih_final_test.py')]},selection_evidence_sha256={str(p):sha256_file(p) for p in [Path('results/model_training/phase25b_reduced_f/run_001/PHASE25B_NEXT_STEP_RECOMMENDATION.md'),Path('results/model_training/phase25b_reduced_f/run_001/FINAL_INTEGRITY.json')]})
    out.mkdir(parents=True,exist_ok=False)
    snapshot(out)
    write_json_new(out/'RR_preservation_before.json',{str(p):sha256_file(p) for p in DEFAULT_RR_ROOT.rglob('*') if p.is_file()})
    write_json_new(out/'pre_evaluation_manifest.json',protocol)
    digest=sha256_file(out/'pre_evaluation_manifest.json')
    (out/'pre_evaluation_manifest.sha256').write_text(digest+'\n')
    (out/'pre_evaluation_manifest.json').chmod(0o444);(out/'pre_evaluation_manifest.sha256').chmod(0o444)
    return index,split,model,state,protocol,digest


class AuthorizedTestIO:
    """Narrow audit allowlist after immutable global claim, denied after exit."""
    def __init__(self,guard,index,ids,claim):
        self.blocked=set(guard.blocked);self.allow=set();self.opens=[];self.denied=[];self.active=False
        self.claim=Path(claim)
        for rid in ids:
            self.allow.add(str((index.root/index.by_record[rid].file).resolve()))
            self.allow.update(str((DEFAULT_DATA_DIR/rid).with_suffix('.'+ext).resolve()) for ext in ('hea','atr'))
    def hook(self,event,args):
        if event!='open':return
        try:path=str(Path(os.fsdecode(os.fspath(args[0]))).resolve())
        except (TypeError,ValueError,OSError):return
        if path not in self.blocked:return
        mode=args[1];flags=args[2]
        writing=(isinstance(mode,str) and any(s in mode for s in 'wax+')) or (isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)))
        if not self.active or path not in self.allow or writing:
            self.denied.append(path);raise PermissionError('Final-test file outside authorized read-only evaluation scope')
        self.opens.append(path)
    def __enter__(self):
        if not self.claim.is_file():raise PermissionError('Exclusive evaluation claim required')
        self.active=True;sys.addaudithook(self.hook);return self
    def __exit__(self,*args):self.active=False


def execute_once(out,index,split,model,state,protocol,digest,old_guard):
    if sha256_file(out/'pre_evaluation_manifest.json')!=digest:raise ValueError('Immutable protocol changed')
    claim_once(ROOT/'evaluation_claim.json',dict(status='single attempt claimed; blocks all further inference',run=str(out),checkpoint_sha256=EXPECTED_SHA,protocol_sha256=digest,time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),historical_QC_incident_acknowledged=True))
    ids=split['summary']['test']['record_ids'];entries=[index.by_record[rid] for rid in ids]
    gate=AuthorizedTestIO(old_guard,index,ids,ROOT/'evaluation_claim.json')
    started=time.perf_counter();prepared={};checks=[]
    torch.manual_seed(42);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    (out/'test_rr_features').mkdir();(out/'predictions').mkdir()
    import wfdb
    try:
        with gate:
            for e in entries:
                a=index.load_shard(e.record_id,split,'test',allow_final_test=True)
                config=json.loads(str(a['preprocessing_config']))
                if ecg_contract(config)!=protocol['ECG_method_contract']:raise ValueError('Test ECG preprocessing method differs')
                n=config['normalization']
                if n['source_ids']!=[e.record_id] or n['channel_names']!=a['channel_names'].tolist() or not np.isfinite(n['offset']).all() or not np.isfinite(n['scale']).all() or np.any(np.asarray(n['scale'])<=0):raise ValueError('Record normalization identity differs')
                base=DEFAULT_DATA_DIR/e.record_id
                for ext in ('hea','atr'):
                    path=base.with_suffix('.'+ext)
                    if sha256_file(path)!=index.manifest['source_files_sha256'][path.name]:raise ValueError('Original test annotation/header checksum differs')
                header=wfdb.rdheader(str(base));ann=wfdb.rdann(str(base),'atr')
                if header.fs!=360 or list(header.sig_name)!=list(e.channel_names):raise ValueError('Header/source contract differs')
                pos=np.asarray(ann.sample,dtype=np.int64);symbols=np.asarray(ann.symbol)
                np.testing.assert_array_equal(pos[a['annotation_indices']],a['annotation_positions']);np.testing.assert_array_equal(symbols[a['annotation_indices']],a['original_symbols'])
                hb=pos[np.asarray([s in BEAT_SYMBOLS for s in ann.symbol])]
                raw,valid=rr_features(hb,a['annotation_positions'],header.fs,RRFeatureConfig())
                rr=transform_rr(raw,valid,state);stable=beat_ids(e.record_id,a['annotation_indices'],a['annotation_positions'])
                with (out/'test_rr_features'/f'{e.record_id}_rr.npz').open('xb') as f:
                    np.savez_compressed(f,RR=rr,rr_raw=raw,rr_valid=valid,beat_ids=stable,annotation_positions=a['annotation_positions'],annotation_indices=a['annotation_indices'],record_ids=a['record_ids'],patient_groups=a['patient_groups'],feature_names=np.asarray(RR_FEATURE_NAMES),source_rows=np.arange(e.count,dtype=np.int64),global_source_indices=e.global_start+np.arange(e.count,dtype=np.int64))
                prepared[e.record_id]=(a,rr,stable)
                checks.append(dict(record_id=e.record_id,patient_group=e.patient_group,count=e.count,source_X_sha256=e.sha256,RR_sha256=sha256_file(out/'test_rr_features'/f'{e.record_id}_rr.npz'),missing_counts=(~valid).sum(0).tolist(),channel_names=list(e.channel_names),finite=True,source_annotation_alignment=True,sampling_frequency=header.fs))
            counts=np.sum([np.bincount(prepared[e.record_id][0]['y'],minlength=5) for e in entries],axis=0)
            np.testing.assert_array_equal(counts,[19246,575,1402,363,1806])
            write_json_new(out/'RR_preparation_integrity.json',dict(records=checks,counts=counts.tolist(),beats=23392,groups=9,scaler_fitted=False,transform_sha256=sha256_file(DEFAULT_RR_ROOT/'training_transform.json'),causal_RR=True,per_record_reset=True))
            engine=SinglePassEngine(model,{e.record_id:e.count for e in entries})
            for e in entries:
                a,rr,stable=prepared.pop(e.record_id)
                write_json_new(out/'predictions'/f'{e.record_id}_inference_started.json',dict(record_id=e.record_id,one_pass=True,count=e.count))
                p=engine.record(e.record_id,a['X'],rr)
                with (out/'predictions'/f'{e.record_id}_predictions.npz').open('xb') as f:
                    np.savez_compressed(f,probabilities=p,y=a['y'],predicted_indices=p.argmax(1).astype(np.int64),record_ids=a['record_ids'],patient_groups=a['patient_groups'],original_symbols=a['original_symbols'],annotation_positions=a['annotation_positions'],annotation_indices=a['annotation_indices'],beat_ids=stable,global_source_indices=e.global_start+np.arange(e.count,dtype=np.int64))
                print(f"Final inference completed once: {e.record_id}, {e.count} beats",flush=True)
            timing=engine.finish()
        if sha256_file(CHECKPOINT)!=EXPECTED_SHA:raise ValueError('Checkpoint changed')
        write_json_new(out/'inference_timing.json',dict(**timing,total_authorized_seconds=time.perf_counter()-started))
        write_json_new(ROOT/'evaluation_completed.json',dict(run=str(out),status='completed and locked; no further model selection or test inference',beats=23392,groups=9,checkpoint_sha256=EXPECTED_SHA,parameters_unchanged=True))
        (ROOT/'evaluation_completed.json').chmod(0o444)
    finally:
        write_json_new(out/'authorized_IO_audit.json',dict(allowed_open_events=gate.opens,denied_open_events=gate.denied,raw_dat_reads=0,scope='Only nine authorized test NPZ + ATR/HEA; no writes to source; disabled after inference',historical_QC_incident_acknowledged=True))
