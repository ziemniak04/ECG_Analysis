"""Phase-18 Checkpoint A: bounded CPU throughput probes and read-only planning.

Never calls train_model, never completes a full epoch, never scores fold 10.
Benchmark models are temporary; full-training configuration is an artifact only.
"""
from dataclasses import asdict
import json
import logging
import math
from pathlib import Path
import shutil
import threading
import time
import os
import platform
import sys
import numpy as np
import torch
from .models.ptbxl_cnn import PTBXLRhythmCNN
from .ptbxl_training import (TrainingConfig,build_plan,make_loader,epoch_pass,
    seed_everything,load_checkpoint,save_checkpoint_new,write_json_new)
from .ptbxl_protocol import PTBXLExportIndex,training_class_counts,training_class_weights,file_sha256

LOG=logging.getLogger(__name__)
EXPECTED_TRAIN_COUNTS=[13354,1189,643,503]


def full_configuration(batch_size=32,cpu_threads=1):
    """Full selection, especially the COMPLETE fold-9 validation, without execution."""
    return TrainingConfig(mode='full',epochs=15,batch_size=batch_size,learning_rate=.001,
        seed=42,device='cpu',cpu_threads=cpu_threads,num_workers=0,train_per_class=None,
        validation_per_class=None,max_train_batches=None,max_validation_batches=None,
        class_weighted_loss=True,patience=4,min_delta=0.)


def validate_full_configuration(config):
    if config.mode!='full' or config.train_per_class is not None or config.max_train_batches is not None:
        raise ValueError('Full training must use all original training-fold records without a batch cap')
    if config.validation_per_class is not None or config.max_validation_batches is not None:
        raise ValueError('Model selection must use the COMPLETE validation fold, without quotas or batch caps')
    if config.device!='cpu':raise ValueError('This readiness benchmark is CPU-only')


def memory_snapshot():
    """Linux RSS/high-water marks, no optional psutil dependency or host writes."""
    values={}
    path=Path('/proc/self/status')
    if path.exists():
        for line in path.read_text().splitlines():
            parts=line.split()
            if parts and parts[0] in ('VmRSS:','VmHWM:'):
                values[parts[0].rstrip(':')]=int(parts[1])*1024
    return {'rss_bytes':values.get('VmRSS'),'process_high_water_bytes':values.get('VmHWM')}


def system_resources(path=Path('.')):
    memory={}
    if Path('/proc/meminfo').exists():
        for line in Path('/proc/meminfo').read_text().splitlines():
            fields=line.split()
            if fields[0] in ('MemTotal:','MemAvailable:','SwapFree:'):
                memory[fields[0].rstrip(':')]=int(fields[1])*1024
    cpu_model=None
    if Path('/proc/cpuinfo').exists():
        cpu_model=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                        if line.startswith('model name')),None)
    return {'memory':memory,'disk_free_bytes':shutil.disk_usage(path).free,
        'logical_cpu_count':os.cpu_count(),'cpu_model':cpu_model,
        'affinity_cpu_count':len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
        'load_average':list(os.getloadavg()) if hasattr(os,'getloadavg') else None,
        'cgroup_cpu_limit':Path('/sys/fs/cgroup/cpu.max').read_text().strip() if Path('/sys/fs/cgroup/cpu.max').exists() else None}


class RSSSampler:
    """Sample current process RSS during probes; process lifetime HWM is separate."""
    def __init__(self):self.peak=0;self.samples=0;self.stop=threading.Event()
    def sample(self):
        value=memory_snapshot()['rss_bytes']
        if value is not None:self.peak=max(self.peak,value);self.samples+=1
    def __enter__(self):
        self.sample()
        def work():
            while not self.stop.wait(.02):self.sample()
        self.thread=threading.Thread(target=work,daemon=True);self.thread.start()
        return self
    def __exit__(self,*args):
        self.stop.set();self.thread.join();self.sample()


class TimedBatches:
    """Profile loading + actual trainer computation for exactly n batches.

    epoch_pass consumes this whole finite iterator. Timings exclude its final
    epoch-level metric reduction, which is measured separately as tail overhead.
    """
    def __init__(self,iterator,count):
        self.iterator=iterator;self.count=count;self.timings=[]
    def __len__(self):return self.count
    def __iter__(self):
        for _ in range(self.count):
            start=time.perf_counter()
            try:batch=next(self.iterator)
            except StopIteration as exc:raise ValueError('Insufficient records for the bounded probe') from exc
            loaded=time.perf_counter()
            yield batch
            end=time.perf_counter()
            self.timings.append({'records':len(batch['y']),'load_seconds':loaded-start,
                'compute_seconds':end-loaded,'end_to_end_seconds':end-start})


def summarize_timings(timings):
    if not timings or any(r['records']<=0 or r['end_to_end_seconds']<=0 for r in timings):
        raise ValueError('Positive measured timings and record counts required')
    elapsed=sum(r['end_to_end_seconds'] for r in timings)
    counts=sum(r['records'] for r in timings)
    values=np.asarray([r['end_to_end_seconds'] for r in timings])
    return {'batches':len(timings),'records':counts,'measured_batch_seconds':values.tolist(),
        'mean_batch_seconds':float(values.mean()),'median_batch_seconds':float(np.median(values)),
        'min_batch_seconds':float(values.min()),'max_batch_seconds':float(values.max()),
        'records_per_second':counts/elapsed,'total_batch_seconds':elapsed,
        'load_seconds':sum(r['load_seconds'] for r in timings),
        'compute_seconds':sum(r['compute_seconds'] for r in timings)}


def project_epoch(train_count,val_count,batch_size,train,val):
    """Planning extrapolation, not a confidence interval or measured full epoch."""
    if min(train_count,val_count,batch_size)<1:raise ValueError('Positive epoch sizes required')
    nt=math.ceil(train_count/batch_size);nv=math.ceil(val_count/batch_size)
    def tail(probe,total):
        # Approximate linear/sorting growth; fixed setup also gets scaled, conservatively.
        samples=probe['records']
        return probe['metric_tail_seconds']*(total/samples)*max(1.,math.log2(max(total,2))/math.log2(max(samples,2)))
    overhead=tail(train,train_count)+tail(val,val_count)+train['loader_setup_seconds']+val['loader_setup_seconds']
    nominal=nt*train['mean_batch_seconds']+nv*val['mean_batch_seconds']+overhead
    lower=nt*train['min_batch_seconds']+nv*val['min_batch_seconds']+overhead
    upper=(nt*train['max_batch_seconds']+nv*val['max_batch_seconds']+overhead)*1.25
    return {'training_batches':nt,'validation_batches':nv,'estimated_train_seconds':nt*train['mean_batch_seconds'],
        'estimated_validation_seconds':nv*val['mean_batch_seconds'],'estimated_metrics_setup_seconds':overhead,
        'estimated_epoch_seconds':nominal,'planning_range_seconds':[lower,upper],
        'fifteen_epoch_nominal_seconds':15*nominal,'fifteen_epoch_upper_planning_seconds':15*upper,
        'method':'ceil(N/B)*observed mean full-batch wall time + extrapolated metrics/setup; last partial batch treated as full',
        'uncertainty':'Observed min/max batch extrapolation with 25% upper planning margin; NOT a statistical confidence interval. Warm caches, load drift, full metric growth, checkpoint writes and startup can differ.',
        'checkpoint_and_startup_time_included':False}


def benchmark_case(index,batch_size,threads,*,timed_batches=6,warmup_batches=1):
    """Limited updates in RAM only; quality metrics are discarded, not selected."""
    if not 1<=timed_batches<=8 or not 0<=warmup_batches<=2:
        raise ValueError('Probe limited to 1..8 timed and 0..2 warmup batches')
    if not 1<=batch_size<=64 or not 1<=threads<=4:raise ValueError('Unbounded CPU probe configuration')
    config=full_configuration(batch_size,threads);validate_full_configuration(config)
    for partition in ('train','validation'):
        if (timed_batches+warmup_batches)*batch_size>len(getattr(index.partitions,partition)):
            raise ValueError('Benchmark cannot exhaust a partition or time a partial batch')
    seed_everything(config);before=memory_snapshot()
    with RSSSampler() as sampler:
        model=PTBXLRhythmCNN();optimizer=torch.optim.Adam(model.parameters(),lr=config.learning_rate)
        weights=torch.from_numpy(training_class_weights(index,index.partitions.train))
        phases={}
        for partition in ('train','validation'):
            start=time.perf_counter()
            loader=make_loader(index,getattr(index.partitions,partition),partition,config,1)
            iterator=iter(loader);setup=time.perf_counter()-start
            kwargs={'partition':partition,'weights':weights,
                    'optimizer':optimizer if partition=='train' else None}
            if warmup_batches:
                warm=TimedBatches(iterator,warmup_batches)
                epoch_pass(model,warm,torch.device('cpu'),**kwargs)
            measured=TimedBatches(iterator,timed_batches)
            start=time.perf_counter()
            metrics=epoch_pass(model,measured,torch.device('cpu'),**kwargs)
            wall=time.perf_counter()-start
            assert len(measured.timings)==timed_batches
            summary=summarize_timings(measured.timings)
            summary.update({'metric_tail_seconds':max(0.,wall-summary['total_batch_seconds']),
                'loader_setup_seconds':setup,'phase_wall_seconds':wall,'finite_loss':math.isfinite(metrics['loss']),
                'folds':sorted(set(metrics['strat_folds'])),'ecg_ids':metrics['ecg_ids'],
                'warmup_batches_excluded':warmup_batches,
                'forward_ms_per_record':metrics['inference_ms_per_record'] if partition=='validation' else None,
                'quality_metrics_reported':False})
            phases[partition]=summary
            LOG.info('CPU probe B=%s threads=%s %s: %.2f records/s, RSS peak %.1f MiB',
                batch_size,threads,partition,summary['records_per_second'],sampler.peak/2**20)
        finite=all(bool(torch.isfinite(p).all()) for p in model.parameters())
    assert finite and phases['validation']['folds']==[9]
    return {'batch_size':batch_size,'cpu_threads':threads,'parameter_count':model.parameter_count(),
        'training_optimizer_updates':warmup_batches+timed_batches,'training_completed_epochs':0,
        'train':phases['train'],'validation':phases['validation'],
        'memory_before':before,'memory_after':memory_snapshot(),
        'sampled_peak_rss_bytes':sampler.peak,'rss_samples':sampler.samples,
        'weights_from_training_only':weights.tolist(),'finite_parameters':finite,
        'projection':project_epoch(len(index.partitions.train),len(index.partitions.validation),batch_size,phases['train'],phases['validation'])}


def recommend_case(cases):
    """Throughput only: within 10% of fastest prefer B32, then fewer CPU threads."""
    if not cases:raise ValueError('No measured probes')
    fastest=min(c['projection']['estimated_epoch_seconds'] for c in cases)
    near=[c for c in cases if c['projection']['estimated_epoch_seconds']<=1.10*fastest]
    return min(near,key=lambda c:(c['batch_size']!=32,c['cpu_threads'],c['projection']['estimated_epoch_seconds']))


def storage_projection(output,full_plan,epochs=15,baseline=Path('results/model_training/phase17_baseline')):
    """Measured Phase-17 checkpoint I/O, clearly labeled full-history size estimate.

    Existing trainer repeats accumulated metric history inside each epoch .pt.
    Account for that triangular growth; no fake ECG arrays or full-run results.
    """
    baseline=Path(baseline);first=baseline/'checkpoints/epoch_001.pt';second=baseline/'checkpoints/epoch_002.pt'
    old_plan=(baseline/'selection_manifest.json').read_bytes()
    full_plan_bytes=len(json.dumps(full_plan,sort_keys=True).encode())
    history_delta=second.stat().st_size-first.stat().st_size
    if history_delta<=0:raise ValueError('Cannot infer a positive history increment from existing checkpoints')
    old=json.loads(old_plan)
    old_records=sum(old['partitions'][k]['records'] for k in ('train','validation'))
    new_records=sum(full_plan['partitions'][k]['records'] for k in ('train','validation'))
    increment=math.ceil(history_delta*new_records/old_records)
    base=max(0,first.stat().st_size-history_delta)+max(0,full_plan_bytes-len(old_plan))
    checkpoint_total=epochs*base+epochs*(epochs+1)//2*increment
    epoch_json=max((baseline/f'epoch_{e:03d}_metrics.json').stat().st_size for e in (1,2))
    metrics_total=math.ceil(epoch_json*new_records/old_records)*epochs
    best_bytes=(baseline/'best_model.pt').stat().st_size
    largest=base+epochs*increment
    estimated_total=checkpoint_total+metrics_total+full_plan_bytes+best_bytes+largest
    checkpoint=load_checkpoint(second);probe=Path(output)/'phase17_checkpoint_io_probe.pt'
    def deep_size(value,seen=None):
        seen=set() if seen is None else seen
        if id(value) in seen:return 0
        seen.add(id(value));size=sys.getsizeof(value)
        if isinstance(value,dict):size+=sum(deep_size(k,seen)+deep_size(v,seen) for k,v in value.items())
        elif isinstance(value,(list,tuple)):size+=sum(deep_size(v,seen) for v in value)
        return size
    history_ram=math.ceil(deep_size(checkpoint['history'])/2*new_records/old_records*epochs)
    start=time.perf_counter();save_checkpoint_new(probe,checkpoint);write_seconds=time.perf_counter()-start
    start=time.perf_counter();loaded=load_checkpoint(probe);load_seconds=time.perf_counter()-start
    for key in checkpoint['model_state_dict']:
        torch.testing.assert_close(checkpoint['model_state_dict'][key],loaded['model_state_dict'][key],rtol=0,atol=0)
    return {'observed_phase17_epoch1_bytes':first.stat().st_size,'observed_phase17_epoch2_bytes':second.stat().st_size,
        'observed_extra_history_bytes':history_delta,'full_plan_json_bytes':full_plan_bytes,
        'estimated_last_full_checkpoint_bytes':largest,'estimated_all_epoch_checkpoints_bytes':checkpoint_total,
        'estimated_epoch_metric_json_bytes':metrics_total,'estimated_run_with_temporary_checkpoint_bytes':estimated_total,
        'estimated_full_python_metric_history_bytes':history_ram,
        'recommended_disk_reserve_bytes':max(2**30,2*estimated_total),
        'note':'Estimate scales observed history by record ratio, adds larger real full-cohort plan and triangular retained-history growth. Not measured full checkpoints; reserve includes 2x margin with minimum 1 GiB, logs/plots/filesystem may differ.',
        'measured_io_probe_bytes':probe.stat().st_size,'measured_checkpoint_write_seconds':write_seconds,
        'measured_checkpoint_load_seconds':load_seconds,'probe_reload_identical':True,
        'probe_source':str(second),'probe_source_sha256':file_sha256(second),'probe_sha256':file_sha256(probe),
        'probe_note':'Copy of existing Phase-17 checkpoint ONLY for I/O; not a new model or Phase-18 selected checkpoint.'}


def run_readiness(output_dir=Path('results/model_training/phase18_checkpoint_a'),
                  config_path=Path('configs/ptbxl_phase18_full_v1.json'),*,timed_batches=6,warmup_batches=1):
    output=Path(output_dir);config_path=Path(config_path)
    targets=['benchmark.json','readiness_plan.json','phase17_checkpoint_io_probe.pt']
    if config_path.exists() or any((output/name).exists() for name in targets):
        raise FileExistsError('Checkpoint-A outputs/configuration already exist; use new non-overwriting paths')
    output.mkdir(parents=True,exist_ok=True)
    resources_before=system_resources();index=PTBXLExportIndex()
    assert (len(index.partitions.train),len(index.partitions.validation),len(index.partitions.test))==(15689,1958,1958)
    counts=training_class_counts(index,index.partitions.train)
    if counts.tolist()!=EXPECTED_TRAIN_COUNTS:raise ValueError('Training counts differ from verified cohort')
    cases=[]
    print(f'Checkpoint A only: four probes, {warmup_batches} warmup + {timed_batches} timed batches/phase, no full epoch, no test loader.',flush=True)
    for batch_size in (16,32):
        for threads in (1,2):
            print(f'Benchmark CPU: batch={batch_size}, threads={threads}, train/validation separately',flush=True)
            cases.append(benchmark_case(index,batch_size,threads,timed_batches=timed_batches,warmup_batches=warmup_batches))
    chosen=recommend_case(cases);config=full_configuration(chosen['batch_size'],chosen['cpu_threads'])
    plan=build_plan(index,config)
    assert plan['partitions']['train']['records']==15689 and plan['partitions']['validation']['records']==1958
    assert plan['partitions']['validation']['batches_per_epoch']==math.ceil(1958/config.batch_size)
    storage=storage_projection(output,plan,config.epochs)
    resources_after=system_resources()
    peak=max(c['sampled_peak_rss_bytes'] for c in cases)
    memory_available=resources_after['memory'].get('MemAvailable')
    report={'checkpoint':'A','full_training_started':False,'approval_required':True,
        'protocol':{'train_records':15689,'validation_records':1958,'test_records_metadata_only':1958,
            'train_patients':len(plan['partitions']['train']['patient_ids']),
            'validation_patients':len(plan['partitions']['validation']['patient_ids']),
            'patient_overlap':False,'test_waveforms_used':False,'training_class_counts':counts.tolist(),
            'training_class_weights':training_class_weights(index,index.partitions.train).tolist()},
        'benchmark_cases':cases,'recommended_batch_size':config.batch_size,'recommended_cpu_threads':config.cpu_threads,
        'recommended_full_configuration':asdict(config),'recommended_projection':chosen['projection'],
        'selection_policy':'Among estimates within 10% of fastest, prefer batch 32 then fewer threads; throughput only, no quality selection',
        'resources_before':resources_before,'resources_after':resources_after,'sampled_peak_rss_bytes':peak,
        'memory_2x_headroom_observed':memory_available>=2*peak if memory_available is not None else None,
        'estimated_full_process_rss_budget_bytes':peak+storage['estimated_full_python_metric_history_bytes'],
        'memory_2x_headroom_estimated':memory_available>=2*(peak+storage['estimated_full_python_metric_history_bytes']) if memory_available is not None else None,
        'disk_reserve_fits':resources_after['disk_free_bytes']>=storage['recommended_disk_reserve_bytes'],
        'storage':storage,'checkpoint_resume_policy':'New output directory, exact config/code/manifest match, last completed epoch; replay partial epoch',
        'environment':{'python':sys.version,'platform':platform.platform(),'torch':str(torch.__version__),'numpy':np.__version__},
        'benchmark_note':'Temporary models discarded; quality metrics from limited probes are not reported or used for selection. Warm-cache measurements and range are planning estimates, not a performance confidence interval.',
        'benchmark_optimizer_updates_total':sum(c['training_optimizer_updates'] for c in cases),
        'benchmark_full_epochs_completed':0}
    write_json_new(output/'readiness_plan.json',plan)
    config_path.parent.mkdir(parents=True,exist_ok=True);write_json_new(config_path,asdict(config))
    write_json_new(output/'benchmark.json',report)
    return report
