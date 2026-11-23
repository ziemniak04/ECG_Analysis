"""Bounded real CPU benchmark, resource inventory, and frozen-protocol checks.

No performance metrics, model selection, full epochs or final-test waveforms.
Benchmark models/optimizer updates are disposable and never exported as candidates.
"""
from dataclasses import asdict, replace
import io
import json
import math
from pathlib import Path
import resource
import shutil
import time
import numpy as np
import torch
from .mitbih_protocol import MITBIHExportIndex, load_split, validate_split, DEFAULT_SPLIT
from .mitbih_torch import MITBIHBeatDataset
from .models.mitbih_cnn import MITBIHBeatCNN
from .mitbih_training import MITBIHTrainingConfig, loss_weights, make_loader, build_plan
from .ptbxl_training import seed_everything, write_json_new
from .prepare_dataset import sha256_file

FROZEN_SPLIT_SHA256 = '8c5cc28d8dcae170fd222432d9f247fd19207ba01be60763241da4b7cae61831'
EXPECTED_COUNTS = {'train':[52254,1715,4389,401,4167],
                   'validation':[19094,491,1444,38,2067], 'test':[19246,575,1402,363,1806]}


def verify_frozen_protocol(index, split):
    summary = validate_split(index, split)
    if sha256_file(DEFAULT_SPLIT) != FROZEN_SPLIT_SHA256 or load_split(index) != split:
        raise ValueError('Phase-20 frozen split changed')
    for part, expected in EXPECTED_COUNTS.items():
        if [summary[part]['class_counts'][c] for c in 'NSVFQ'] != expected:
            raise ValueError('Frozen partition counts differ')
    pair = index.by_record['201'].patient_group
    if pair != index.by_record['202'].patient_group or pair not in split['groups']['validation']:
        raise ValueError('201/202 must be together in validation')
    return summary


def resources():
    values = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        if key in ('MemTotal','MemAvailable','SwapFree'):
            values[key+'_bytes'] = int(value.split()[0])*1024
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith(('VmRSS:', 'VmHWM:')):
            key, value = line.split(':',1); values[key+'_bytes'] = int(value.split()[0])*1024
    values['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    values['disk_free_bytes'] = shutil.disk_usage('.').free
    return values


def cached_bytes(dataset):
    return sum(a.nbytes for arrays, _ in dataset.cache.values() for a in arrays.values())


def preload(dataset):
    """One cold loader-cache pass, bounded to this partition; never read test."""
    if dataset.partition not in ('train','validation'):
        raise PermissionError('No final-test benchmark')
    start = time.perf_counter(); offset = 0
    for entry in dataset.shards:
        dataset[offset]; offset += entry.count
    return dict(seconds=time.perf_counter()-start, shards=len(dataset.shards),
                cache_bytes=cached_bytes(dataset), cache_entries=len(dataset.cache))


def benchmark(output, *, timed_batches=6, warmup_batches=2, batch_sizes=(16,32,64)):
    """At most 8 training and 8 inference batches per size; no full-training call."""
    if (not 1 <= timed_batches <= 6 or not 0 <= warmup_batches <= 2
            or any(b not in (16,32,64) for b in batch_sizes) or len(set(batch_sizes)) != len(batch_sizes)):
        raise ValueError('Benchmark must remain short and use candidate sizes 16/32/64')
    output = Path(output)
    if output.exists():
        raise FileExistsError('Use a new benchmark directory')
    index = MITBIHExportIndex(); split = load_split(index)
    summary = verify_frozen_protocol(index, split)
    config = MITBIHTrainingConfig()
    # Cache is bounded to 29 TRAIN and 10 VALIDATION shards, not the full dataset.
    train = MITBIHBeatDataset(index, split, partition='train', max_cached_shards=29)
    validation = MITBIHBeatDataset(index, split, partition='validation', max_cached_shards=10)
    before = resources(); output.mkdir(parents=True, exist_ok=False)
    seed_everything(config)
    cold_train = preload(train); cold_val = preload(validation)
    after_load = resources()
    weights = torch.from_numpy(loss_weights(index, split, config.loss_strategy))
    results = []
    for size in batch_sizes:
        current = replace(config, batch_size=size); seed_everything(current)
        model = MITBIHBeatCNN(); optimizer = torch.optim.Adam(model.parameters(), lr=current.learning_rate)
        criterion = torch.nn.CrossEntropyLoss(weight=weights)
        timings = {'train':[], 'validation':[]}; total_updates = 0
        for part, dataset in [('train', train), ('validation', validation)]:
            model.train(part == 'train'); loader = iter(make_loader(dataset, current, 1))
            context = torch.enable_grad() if part == 'train' else torch.inference_mode()
            with context:
                for step in range(warmup_batches+timed_batches):
                    start = time.perf_counter(); batch = next(loader); loaded = time.perf_counter()
                    if not set(batch['patient_group']) <= set(split['groups'][part]):
                        raise RuntimeError('Wrong group in benchmark')
                    if part == 'train': optimizer.zero_grad(set_to_none=True)
                    tick = time.perf_counter(); logits = model(batch['X']); forward = time.perf_counter()-tick
                    loss = criterion(logits, batch['y'])
                    if not bool(torch.isfinite(loss)): raise RuntimeError('Nonfinite benchmark loss')
                    if part == 'train':
                        loss.backward()
                        if not all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                            raise RuntimeError('Invalid benchmark gradients')
                        optimizer.step(); total_updates += 1
                    end = time.perf_counter()
                    if step >= warmup_batches:
                        timings[part].append(dict(loading_seconds=loaded-start, compute_seconds=end-loaded,
                            forward_seconds=forward, total_seconds=end-start, beats=len(batch['y'])))
        stats = {}
        for part, rows in timings.items():
            total = sum(r['total_seconds'] for r in rows); beats = sum(r['beats'] for r in rows)
            per_batch = total/len(rows)
            stats[part] = dict(timed_batches=len(rows), beats=beats, beats_per_second=beats/total,
                batch_seconds=per_batch, loading_seconds=sum(r['loading_seconds'] for r in rows),
                compute_seconds=sum(r['compute_seconds'] for r in rows),
                forward_seconds=sum(r['forward_seconds'] for r in rows),
                min_batch_seconds=min(r['total_seconds'] for r in rows),
                max_batch_seconds=max(r['total_seconds'] for r in rows),
                estimated_full_pass_seconds=math.ceil(summary[part]['beats']/size)*per_batch)
        buffer = io.BytesIO()
        torch.save(dict(model_state_dict=model.state_dict(), best_model_state_dict={k:v.clone() for k,v in model.state_dict().items()},
                        optimizer_state_dict=optimizer.state_dict(), rng_state=torch.get_rng_state()), buffer)
        results.append(dict(batch_size=size, stages=stats, updates_disposable_model=total_updates,
            checkpoint_tensor_payload_bytes=buffer.tell(), resources=resources(),
            estimated_warm_epoch_seconds=sum(s['estimated_full_pass_seconds'] for s in stats.values()),
            estimated_first_epoch_seconds=sum(s['estimated_full_pass_seconds'] for s in stats.values())+cold_train['seconds']+cold_val['seconds']))
        print(json.dumps(results[-1]), flush=True)
    best = min(results, key=lambda r:r['estimated_warm_epoch_seconds'])
    chosen = replace(config, batch_size=best['batch_size'])
    result = dict(schema='mitbih-cpu-readiness-v1', status='passed', configuration=asdict(chosen),
        summary=summary, resources_before=before, resources_after_preload=after_load,
        resources_after=resources(), cold_loader_cache_train=cold_train, cold_loader_cache_validation=cold_val,
        benchmark_rows=results, recommended_batch_size=best['batch_size'],
        measured_training_updates=sum(r['updates_disposable_model'] for r in results),
        final_test_waveforms_read=False, full_training_called=False,
        read_records=index.read_records, model_performance_metrics_computed=False,
        uncertainty='Short warm-cache CPU sample; first pass adds measured shard load. OS cache was not flushed. '
            'Epoch estimates omit full-population metric sorting, checkpoints, logging and scheduling variability. '
            'A 0.5x-2x planning range is heuristic, not a confidence interval.')
    assert set(index.read_records) <= set(summary['train']['record_ids']+summary['validation']['record_ids'])
    write_json_new(output/'benchmark.json', result)
    write_json_new(output/'training_plan.json', build_plan(index, split, chosen))
    return result
