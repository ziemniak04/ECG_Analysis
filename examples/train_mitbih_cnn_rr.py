"""Approved Phase24B single full experiment, guarded before heavy imports."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.mitbih_execution_guard import HoldoutAccessGuard

ROOT=Path('results/model_training/phase24b_mitbih_cnn_rr')
if __name__=='__main__':
    with HoldoutAccessGuard(ROOT/'runtime_guard_training_001.json') as guard:
        probe=guard.probe()
        import json,time,platform,importlib.metadata
        import numpy as np
        import wfdb
        from dataclasses import asdict
        from src.mitbih_cnn_rr_training import MITBIHTrainingConfig,train_model,build_plan
        from src.mitbih_full_run import snapshot,verify_config
        from src.mitbih_protocol import MITBIHExportIndex,load_split
        from src.mitbih_readiness import verify_frozen_protocol,resources
        from src.mitbih_rr_dataset import MITBIHRRDataset,DEFAULT_RR_ROOT
        from src.mitbih_rr import rr_features,fit_training_transform
        from src.visualization import BEAT_SYMBOLS
        from src.data_loader import DEFAULT_DATA_DIR
        from src.mitbih_unweighted import BASELINE,BASELINE_SHA256
        from src.prepare_dataset import sha256_file
        from src.ptbxl_training import write_json_new
        output=ROOT/'run_001';audit=ROOT/'preflight_001'
        if output.exists() or audit.exists():raise FileExistsError('Preserve existing execution; inspect before resume')
        config=MITBIHTrainingConfig(**json.loads(Path('PHASE21_MITBIH_TRAINING_CONFIG.json').read_text()))
        verify_config(config)
        if asdict(config)!=json.loads((BASELINE/'configuration.json').read_text()):raise ValueError('Baseline settings differ')
        if sha256_file(BASELINE/'best_model.pt')!=BASELINE_SHA256:raise ValueError('Baseline changed')
        if sha256_file(DEFAULT_RR_ROOT/'manifest.json')!='43fe395464a0ba187eae01f122b28f96662dca2a770140c39fbc803b6bb6c335':raise ValueError('RR manifest changed')
        index=MITBIHExportIndex();split=load_split(index);summary=verify_frozen_protocol(index,split)
        capacity=resources()
        if capacity['MemAvailable_bytes']<2*2**30 or capacity['disk_free_bytes']<128*2**20:raise RuntimeError('Insufficient resources')
        audit.mkdir(parents=True,exist_ok=False);snapshot(audit)
        rr_protection={str(p):sha256_file(p) for p in DEFAULT_RR_ROOT.rglob('*') if p.is_file()}
        write_json_new(audit/'rr_preservation_before.json',rr_protection)
        raw_train=[];valid_train=[];training_groups=[];verified={}
        for partition in ('train','validation'):
            ds=MITBIHRRDataset(index=index,split=split,partition=partition,max_cached_shards=1)
            offset=0;counts=np.zeros(5,dtype=np.int64)
            for e in ds.base.shards:
                sample=ds[offset]  # verifies entire original X and RR metadata/values, not just first row
                row=ds.rows[e.record_id]
                with np.load(DEFAULT_RR_ROOT/row['file'],allow_pickle=False) as z:
                    a={k:z[k] for k in z.files}
                base=DEFAULT_DATA_DIR/e.record_id
                header=wfdb.rdheader(str(base));ann=wfdb.rdann(str(base),'atr')
                positions=np.asarray(ann.sample,dtype=np.int64)
                hb=positions[np.asarray([s in BEAT_SYMBOLS for s in ann.symbol])]
                raw,valid=rr_features(hb,a['annotation_positions'],header.fs)
                np.testing.assert_array_equal(raw,a['rr_raw']);np.testing.assert_array_equal(valid,a['rr_valid'])
                np.testing.assert_array_equal(positions[a['annotation_indices']],a['annotation_positions'])
                np.testing.assert_array_equal(np.asarray(ann.symbol)[a['annotation_indices']],a['original_symbols'])
                if e.record_id in ('201','202') and a['rr_valid'][0].any():raise ValueError('Cross-record context')
                counts+=np.bincount(a['y'],minlength=5)
                if partition=='train':raw_train.append(raw);valid_train.append(valid);training_groups.extend([e.patient_group]*len(raw))
                verified[e.record_id]=dict(partition=partition,count=len(raw),RR_sha256=sha256_file(DEFAULT_RR_ROOT/row['file']),X_sha256=e.sha256)
                offset+=e.count
            if counts.tolist()!=[summary[partition]['class_counts'][c] for c in 'NSVFQ']:raise ValueError('Class alignment')
        independent=fit_training_transform(np.concatenate(raw_train),np.concatenate(valid_train),partition='train',patient_groups=training_groups,expected_training_groups=split['groups']['train'],manifest_sha256=index.manifest_sha256)
        if independent!=ds.transform:raise ValueError('Training-only transform differs')
        plan=build_plan(index,split,config)
        if guard.unexpected:raise RuntimeError('Unexpected held-out dependency')
        write_json_new(audit/'preflight.json',dict(status='passed',guard_probe=probe,resources=capacity,summary=summary,verified_records=verified,training_only_transform_independently_verified=True,baseline_sha256=BASELINE_SHA256,parameter_count=17989,configuration=asdict(config),historical_24A_incident_acknowledged=True))
        write_json_new(audit/'reproducibility.json',dict(python=sys.version,platform=platform.platform(),dependencies={d.metadata['Name']:d.version for d in importlib.metadata.distributions() if d.metadata['Name']},seed=42,cpu_threads=2,model_initialization='fresh; no learned checkpoint loaded',approval='User explicitly approved exactly one Phase24B full CNN+RR experiment'))
        del raw_train,valid_train,ds,raw,valid,a
        print(json.dumps({'preflight':'passed','configuration':asdict(config),'resources':capacity,'output':str(output),'guard_probe':probe},indent=2),flush=True)
        start=time.perf_counter()
        result=train_model(config,output,index=index,split=split,allow_full_training=True)
        allowed=set(summary['train']['record_ids']+summary['validation']['record_ids'])
        if not set(index.read_records)<=allowed or guard.unexpected:raise RuntimeError('Isolation failed')
        write_json_new(output/'execution.json',dict(wall_seconds=time.perf_counter()-start,result=result,read_records=index.read_records,unexpected_heldout_reads=0,resources_after=resources()))
        print(json.dumps(result,indent=2),flush=True)
