"""Approved Phase-18 execution/audit wrapper around the unchanged Phase-17 trainer.

Timing hooks delegate all numerical work unchanged. Guarded reads prohibit fold
10. Managed status is atomically updated; all experiment outputs are new files.
"""
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime,timezone
import csv
import json
import logging
import os
from pathlib import Path
import signal
import time
from uuid import uuid4
import numpy as np
import torch
from . import ptbxl_training as trainer
from .ptbxl_protocol import PTBXLExportIndex,training_class_counts,file_sha256
from .ptbxl_model_config import RhythmCNNConfig
from .ptbxl_readiness import validate_full_configuration,system_resources

APPROVED={'mode':'full','epochs':15,'batch_size':16,'learning_rate':.001,'seed':42,
 'device':'cpu','cpu_threads':2,'num_workers':0,'train_per_class':None,'validation_per_class':None,
 'max_train_batches':None,'max_validation_batches':None,'class_weighted_loss':True,'patience':4,'min_delta':0.}


def utc_now():return datetime.now(timezone.utc).isoformat()


def atomic_status(audit,**value):
    """Only this run's live status is mutable; historical files are never replaced."""
    audit=Path(audit);temporary=audit/('.status-'+uuid4().hex+'.json')
    with temporary.open('x') as target:
        json.dump({'updated_utc':utc_now(),**value},target,indent=2,allow_nan=False)
        target.flush();os.fsync(target.fileno())
    os.replace(temporary,audit/'live_status.json')


class GuardedIndex(PTBXLExportIndex):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.read_counts={};self.blocked_test_reads=0
    def load_numpy(self,position,*,verify_checksum=True):
        if self.entries[position].strat_fold==10:
            self.blocked_test_reads+=1
            raise RuntimeError('Final-test waveform access is prohibited in Phase 18')
        fold=self.entries[position].strat_fold
        self.read_counts[fold]=self.read_counts.get(fold,0)+1
        return super().load_numpy(position,verify_checksum=verify_checksum)


@contextmanager
def epoch_timing(audit,output,index):
    """Capture full epochs from loader setup to persisted checkpoint/log completion."""
    audit,output=Path(audit),Path(output)
    original_loader,original_pass=trainer.make_loader,trainer.epoch_pass
    starts={};phases={};active=[None]
    def loader(*args,**kwargs):
        partition=args[2] if len(args)>2 else kwargs['partition']
        epoch=args[4] if len(args)>4 else kwargs['epoch']
        if partition=='train':
            active[0]=epoch;starts[epoch]=(time.perf_counter(),utc_now());phases[epoch]={}
            atomic_status(audit,status='training',epoch=epoch,output_dir=str(output),
                          read_counts=index.read_counts,blocked_test_reads=index.blocked_test_reads)
        return original_loader(*args,**kwargs)
    def epoch_pass(*args,**kwargs):
        start=time.perf_counter()
        result=original_pass(*args,**kwargs)
        phases[active[0]][kwargs['partition']]=time.perf_counter()-start
        return result
    class Completed(logging.Handler):
        def emit(self,record):
            if not str(record.msg).startswith('epoch %s: train loss'):return
            epoch=int(record.args[0]);duration=time.perf_counter()-starts[epoch][0]
            row={'epoch':epoch,'start_utc':starts[epoch][1],'end_utc':utc_now(),
                 'epoch_seconds':duration,'train_seconds':phases[epoch]['train'],
                 'validation_seconds':phases[epoch]['validation'],
                 'setup_checkpoint_reporting_seconds':max(0.,duration-sum(phases[epoch].values()))}
            with (audit/'epoch_times.jsonl').open('a') as target:
                target.write(json.dumps(row)+'\n');target.flush();os.fsync(target.fileno())
            atomic_status(audit,status='epoch_completed',epoch=epoch,output_dir=str(output),
                epoch_seconds=duration,train_loss=float(record.args[1]),validation_loss=float(record.args[2]),
                validation_macro_f1=float(record.args[3]),read_counts=index.read_counts,
                blocked_test_reads=index.blocked_test_reads)
    handler=Completed();trainer.LOG.addHandler(handler)
    trainer.make_loader,trainer.epoch_pass=loader,epoch_pass
    try:yield
    finally:
        trainer.make_loader,trainer.epoch_pass=original_loader,original_pass
        trainer.LOG.removeHandler(handler);handler.close()


def preflight(config_path,output,audit):
    config_path,output,audit=map(Path,(config_path,output,audit))
    if output.exists():raise FileExistsError(f'Output exists: {output}; use a NEW version directory')
    config=trainer.TrainingConfig(**json.loads(config_path.read_text()))
    if asdict(config)!=APPROVED:raise ValueError('Configuration differs from explicitly approved Phase-18 settings')
    validate_full_configuration(config)
    audit.mkdir(parents=True,exist_ok=True)
    if (audit/'preflight.json').exists():raise FileExistsError('Preflight audit already exists; use a new audit directory')
    protected=[]
    for folder in ('src','tests','examples','configs'):
        protected.extend(p for p in Path(folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    protected.extend(p for p in Path('.').iterdir() if p.is_file())
    protected.extend(p for p in Path('results/model_training/phase17_baseline').rglob('*') if p.is_file())
    protected.extend([Path('data/processed/ecg_dataset_v1/dataset_metadata.json'),Path('data/processed/ptbxl_rhythm_v1/dataset_identity.json')])
    trainer.write_json_new(audit/'preservation_before.json',{str(p):file_sha256(p) for p in protected})
    dataset_stats={str(p):[p.stat().st_size,p.stat().st_mtime_ns]
        for root in ('data/processed/ecg_dataset_v1','data/processed/ptbxl_rhythm_v1')
        for p in Path(root).rglob('*') if p.is_file()}
    trainer.write_json_new(audit/'dataset_stats_before.json',dataset_stats)
    atomic_status(audit,status='preflight',validated_records=0,output_dir=str(output))
    index=GuardedIndex();plan=trainer.build_plan(index,config)
    if (plan['partitions']['train']['records'],plan['partitions']['validation']['records'])!=(15689,1958):raise ValueError('Cohort counts differ')
    if training_class_counts(index,index.partitions.train).tolist()!=[13354,1189,643,503]:raise ValueError('Training counts differ')
    resources=system_resources()
    if resources['disk_free_bytes']<2**30:raise RuntimeError('Less than approved 1-GiB disk reserve')
    if resources['memory'].get('MemAvailable',0)<2**30:raise RuntimeError('Less than 1-GiB available memory')
    if RhythmCNNConfig().expected_parameter_count()!=74116:raise ValueError('Architecture changed')
    start=time.perf_counter();verified=[]
    selected=(*index.partitions.train,*index.partitions.validation)
    for n,position in enumerate(selected,1):
        arrays=index.load_numpy(position)
        entry=index.entries[position]
        if arrays['X'].shape!=(12,1000) or not np.isfinite(arrays['X']).all():raise ValueError('Invalid ECG array')
        verified.append({'ecg_id':entry.ecg_id,'file':entry.file,'sha256':entry.sha256,'strat_fold':entry.strat_fold})
        if n%1000==0:
            atomic_status(audit,status='preflight',validated_records=n,output_dir=str(output))
            print(f'Preflight: {n}/17647 training+validation NPZ verified',flush=True)
    assert len(verified)==17647 and not index.blocked_test_reads
    trainer.write_json_new(audit/'verified_input_checksums.json',verified)
    result={'status':'passed','config':asdict(config),'config_sha256':file_sha256(config_path),
        'manifest_sha256':index.manifest_sha256,'plan_sha256':trainer._hash(plan),
        'train_records':15689,'validation_records':1958,'patient_overlap':False,
        'test_waveforms_read':0,'verified_npz':len(verified),'preflight_seconds':time.perf_counter()-start,
        'resources':resources,'created_utc':utc_now(),'planned_output':str(output)}
    trainer.write_json_new(audit/'preflight.json',result)
    atomic_status(audit,status='preflight_passed',validated_records=len(verified),output_dir=str(output))
    return result


def _postprocess(output,audit,training_seconds):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output,audit=Path(output),Path(audit)
    result=json.loads((output/'result.json').read_text())
    checkpoint=trainer.load_checkpoint(result['last_completed_checkpoint'])
    history=checkpoint['history']
    durations=[json.loads(line) for line in (audit/'epoch_times.jsonl').read_text().splitlines() if line]
    times={row['epoch']:row for row in durations}
    with (output/'epoch_durations.csv').open('x',newline='') as target:
        writer=csv.DictWriter(target,fieldnames=list(durations[0]));writer.writeheader();writer.writerows(durations)
    fields=['epoch','epoch_seconds','training_loss','validation_loss','training_accuracy','validation_accuracy',
            'training_macro_f1','validation_macro_f1','training_macro_auprc','validation_macro_auprc']
    with (output/'epoch_metrics_full.csv').open('x',newline='') as target:
        writer=csv.DictWriter(target,fieldnames=fields);writer.writeheader()
        for row in history:
            writer.writerow({'epoch':row['epoch'],'epoch_seconds':times.get(row['epoch'],{}).get('epoch_seconds'),
                **{f'{name}_{metric}':row[partition][metric] for name,partition in [('training','train'),('validation','validation')]
                   for metric in ('loss','accuracy','macro_f1','macro_auprc')}})
    per_class_fields=['epoch','partition','class','class_index','support','predicted_count','recall','specificity','precision','f1','auprc']
    with (output/'per_class_epoch_metrics.csv').open('x',newline='') as target:
        writer=csv.DictWriter(target,fieldnames=per_class_fields);writer.writeheader()
        for row in history:
            for partition in ('train','validation'):
                for metrics in row[partition]['per_class']:
                    writer.writerow({'epoch':row['epoch'],'partition':partition,**{key:metrics[key] for key in per_class_fields[2:]}})
    best=json.loads((output/'best_validation_metrics.json').read_text())
    with (output/'best_validation_per_class.csv').open('x',newline='') as target:
        writer=csv.DictWriter(target,fieldnames=list(best['per_class'][0]));writer.writeheader();writer.writerows(best['per_class'])
    with (output/'validation_confusion_matrix.csv').open('x',newline='') as target:
        writer=csv.writer(target);writer.writerow(['true/predicted','SR','AFIB','STACH','SBRAD'])
        for label,row in zip(('SR','AFIB','STACH','SBRAD'),best['confusion_matrix']):writer.writerow([label,*row])
    fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
    ax.plot([row['epoch'] for row in history],[row['validation']['macro_auprc'] for row in history],marker='o')
    ax.set_xlabel('Completed epoch');ax.set_ylabel('Validation Macro AUPRC (average precision)')
    ax.set_title('Full fold-9 validation');ax.set_ylim(0,1);ax.grid(alpha=.25)
    with (output/'validation_macro_auprc.png').open('xb') as target:fig.savefig(target,format='png',dpi=300)
    plt.close(fig)
    os.link(Path(result['last_completed_checkpoint']),output/'last_resumable_checkpoint.pt')
    summary={'status':result['status'],'training_call_seconds':training_seconds,
        'epochs':result['completed_epochs'],'early_stopped':result['status']=='early_stopped',
        'best_epoch':result['best_epoch'],'best_validation_macro_f1':result['best_validation_macro_f1'],
        'last_checkpoint':result['last_completed_checkpoint'],'epoch_durations_recorded':len(times),
        'best_model_sha256':file_sha256(output/'best_model.pt'),
        'last_checkpoint_sha256':file_sha256(output/'last_resumable_checkpoint.pt')}
    trainer.write_json_new(output/'full_run_summary.json',summary)
    return summary


def execute(config_path,output,audit,*,resume_from=None,prior_audit=None):
    output,audit=Path(output),Path(audit)
    pre=json.loads((audit/'preflight.json').read_text())
    config=trainer.TrainingConfig(**json.loads(Path(config_path).read_text()))
    if pre['status']!='passed' or asdict(config)!=APPROVED or file_sha256(config_path)!=pre['config_sha256']:
        raise ValueError('Approved configuration/preflight mismatch')
    if output.exists():raise FileExistsError('New output directory required')
    index=GuardedIndex()
    if index.manifest_sha256!=pre['manifest_sha256'] or trainer._hash(trainer.build_plan(index,config))!=pre['plan_sha256']:
        raise ValueError('Metadata/source changed since preflight')
    if prior_audit:
        previous=Path(prior_audit)/'epoch_times.jsonl'
        with (audit/'epoch_times.jsonl').open('x') as target:target.write(previous.read_text())
    trainer.write_json_new(audit/'worker_started.json',{'pid':os.getpid(),'created_utc':utc_now(),
        'output':str(output),'config':str(config_path),'resume_from':str(resume_from) if resume_from else None})
    previous_handler=signal.getsignal(signal.SIGTERM)
    def terminate(signum,frame):raise KeyboardInterrupt('SIGTERM received; preserve completed checkpoints')
    signal.signal(signal.SIGTERM,terminate)
    start=time.perf_counter();started_utc=utc_now()
    try:
        with epoch_timing(audit,output,index):
            result=trainer.train_model(config,output,index=index,resume_from=resume_from,allow_full_training=True)
        elapsed=time.perf_counter()-start
        if result['last_completed_checkpoint']:
            summary=_postprocess(output,audit,elapsed)
        else:summary={'status':result['status'],'epochs':0,'training_call_seconds':elapsed}
        summary.update({'started_utc':started_utc,'ended_utc':utc_now(),'data_read_counts_by_fold':index.read_counts,
                        'blocked_test_reads':index.blocked_test_reads})
        trainer.write_json_new(audit/'execution_summary.json',summary)
        atomic_status(audit,**summary)
        return summary
    except BaseException as exc:
        atomic_status(audit,status='worker_failed',error=f'{type(exc).__name__}: {exc}',
            output_dir=str(output),read_counts=index.read_counts,blocked_test_reads=index.blocked_test_reads)
        raise
    finally:signal.signal(signal.SIGTERM,previous_handler)
