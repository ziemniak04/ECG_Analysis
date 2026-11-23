"""Controlled Phase-18C loss ablation around the unchanged trainer and model.

Only effective CrossEntropyLoss weights differ. The scoped adapter records its
own fingerprint in resumable plans, so sqrt weights cannot resume as balanced.
It is single-experiment/single-thread orchestration, not a concurrent trainer API.
"""
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import datetime, timezone
import _thread
import json
import logging
from pathlib import Path
import signal
import threading
import time

import numpy as np
import torch

from . import ptbxl_training as trainer
from .models.ptbxl_cnn import PTBXLRhythmCNN
from .ptbxl_full_run import GuardedIndex, epoch_timing, _postprocess, atomic_status
from .ptbxl_protocol import training_class_weights, training_class_counts, file_sha256
from .ptbxl_readiness import system_resources, memory_snapshot

BASELINE = Path('results/model_training/phase18_full/run_001')
VARIANTS = {'B_unweighted': 'none', 'C_sqrt_balanced': 'sqrt_balanced_mean_one'}
COMMON = {'mode': 'full', 'epochs': 15, 'batch_size': 16, 'learning_rate': .001,
          'seed': 42, 'device': 'cpu', 'cpu_threads': 2, 'num_workers': 0,
          'train_per_class': None, 'validation_per_class': None,
          'max_train_batches': None, 'max_validation_batches': None,
          'patience': 4, 'min_delta': 0.}
SELECTION = {
    'rules': [0., .65], 'minimum_AFIB_recall': .90,
    'minimum_STACH_recall': .75, 'minimum_SBRAD_recall': .70,
    'minimum_coverage': .97, 'maximum_auprc_loss_vs_A': .02,
    'macro_f1_tie_band': .005, 'macro_auprc_tie_band': .005,
    'false_positive_tie_band': 5,
    'order': ['eligibility gates', 'within 0.005 of best full-cohort Macro F1',
              'within 0.005 of best full-cohort probability Macro AUPRC',
              'within 5 of minimum SBRAD false positives',
              'prefer argmax, then unweighted, sqrt-balanced, original balanced'],
    'note': 'Engineering tolerances, not significance tests or clinical criteria.'}


def configuration(variant):
    if variant not in VARIANTS:
        raise ValueError('Only the two NEW loss variants may be trained')
    return trainer.TrainingConfig(**COMMON, class_weighted_loss=variant == 'C_sqrt_balanced')


def partial_training_weights(index, indices):
    """Reuse original training-only guards; sqrt(float32 balanced), mean=1."""
    balanced = training_class_weights(index, indices).astype(np.float64)
    weights = np.sqrt(balanced)
    return (weights / weights.mean()).astype(np.float32)


@contextmanager
def controlled_loss(variant):
    """Delegate all optimization, shuffling and checkpointing to the old trainer."""
    if variant not in VARIANTS:
        raise ValueError('Unknown new variant')
    original_plan = trainer.build_plan
    original_weights = trainer.training_class_weights
    def weights(index, indices):
        if variant != 'C_sqrt_balanced':
            raise RuntimeError('Unweighted run must not request loss weights')
        return partial_training_weights(index, indices)
    def plan(index, config):
        if config.class_weighted_loss != (variant == 'C_sqrt_balanced'):
            raise ValueError('Loss scheme/configuration mismatch')
        value = original_plan(index, config)
        indices = tuple(value['partitions']['train']['global_indices'])
        value['loss_ablation'] = {'variant': variant, 'scheme': VARIANTS[variant],
            'balanced_training_weights': training_class_weights(index, indices).tolist(),
            'effective_training_weights': value['class_weights'],
            'source': 'training folds only; never validation or test'}
        value['code_sha256']['ptbxl_optimization.py'] = file_sha256(Path(__file__))
        return value
    trainer.training_class_weights, trainer.build_plan = weights, plan
    try:
        yield
    finally:
        trainer.training_class_weights, trainer.build_plan = original_weights, original_plan


def select_candidate(rows):
    """Prespecified gates and practical tie bands; no coverage-conditional ranking."""
    reference = next(r for r in rows if r['model'] == 'A_balanced' and r['threshold'] == 0)
    eligible = [r for r in rows
        if r['AFIB_recall'] >= SELECTION['minimum_AFIB_recall']
        and r['STACH_recall'] >= SELECTION['minimum_STACH_recall']
        and r['SBRAD_recall'] >= SELECTION['minimum_SBRAD_recall']
        and r['coverage'] >= SELECTION['minimum_coverage']
        and r['macro_auprc_full_cohort'] >= reference['macro_auprc_full_cohort'] - SELECTION['maximum_auprc_loss_vs_A']]
    if not eligible:
        return {'status': 'no_eligible_candidate', 'selected': None, 'criterion': SELECTION}
    stages = {'eligible': [r['candidate'] for r in eligible]}
    best_f1 = max(r['macro_f1_full_cohort'] for r in eligible)
    candidates = [r for r in eligible if r['macro_f1_full_cohort'] >= best_f1 - SELECTION['macro_f1_tie_band']]
    stages['macro_f1_band'] = [r['candidate'] for r in candidates]
    best_ap = max(r['macro_auprc_full_cohort'] for r in candidates)
    candidates = [r for r in candidates if r['macro_auprc_full_cohort'] >= best_ap - SELECTION['macro_auprc_tie_band']]
    stages['auprc_band'] = [r['candidate'] for r in candidates]
    fewest_fp = min(r['SBRAD_fp'] for r in candidates)
    candidates = [r for r in candidates if r['SBRAD_fp'] <= fewest_fp + SELECTION['false_positive_tie_band']]
    stages['false_positive_band'] = [r['candidate'] for r in candidates]
    simplicity = {'B_unweighted': 0, 'C_sqrt_balanced': 1, 'A_balanced': 2}
    selected = min(candidates, key=lambda r: (r['threshold'] != 0, simplicity[r['model']], r['candidate']))
    return {'status': 'selected', 'selected': selected, 'stages': stages, 'criterion': SELECTION}


class ResourceGuard:
    """Stop at safe trainer boundaries when resources substantially exceed plan."""
    def __init__(self, audit, limits):
        self.audit, self.limits = Path(audit), limits
        self.stop = threading.Event(); self.violation = None
        self.peak_rss = 0; self.minimum_available_ram = None; self.minimum_disk = None
    def sample(self):
        resource = system_resources(); rss = memory_snapshot()['rss_bytes'] or 0
        available = resource['memory'].get('MemAvailable', 0); disk = resource['disk_free_bytes']
        self.peak_rss = max(self.peak_rss, rss)
        self.minimum_available_ram = available if self.minimum_available_ram is None else min(self.minimum_available_ram, available)
        self.minimum_disk = disk if self.minimum_disk is None else min(self.minimum_disk, disk)
        if rss > self.limits['maximum_process_rss_bytes']:
            self.violation = 'Process RSS exceeded the prespecified resource limit'
        elif available < self.limits['minimum_available_ram_bytes']:
            self.violation = 'Available system RAM fell below safe reserve'
        elif disk < self.limits['minimum_disk_bytes']:
            self.violation = 'Disk free space fell below safe reserve'
        status_path = self.audit / 'live_status.json'
        if status_path.exists():
            status = json.loads(status_path.read_text())
            if status.get('status') == 'training':
                elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(status['updated_utc'])).total_seconds()
                if elapsed > self.limits['maximum_epoch_seconds']:
                    self.violation = 'Epoch exceeded twice the slowest measured Phase-18 epoch'
    def __enter__(self):
        self.sample()
        if self.violation:
            raise RuntimeError(self.violation)
        def monitor():
            while not self.stop.wait(10):
                self.sample()
                if self.violation:
                    _thread.interrupt_main(); return
        self.thread = threading.Thread(target=monitor, daemon=True); self.thread.start()
        return self
    def __exit__(self, *args):
        self.stop.set(); self.thread.join(); self.sample()
        trainer.write_json_new(self.audit / 'resource_usage.json', {
            'limits': self.limits, 'violation': self.violation,
            'peak_rss_bytes': self.peak_rss,
            'minimum_available_ram_bytes': self.minimum_available_ram,
            'minimum_disk_free_bytes': self.minimum_disk})


def prepare(root):
    """Immutable plan/configs, baseline hashes, all train/validation NPZ integrity."""
    root = Path(root)
    if root.exists():
        raise FileExistsError('Use a new versioned optimization directory')
    root.mkdir(parents=True, exist_ok=False)
    protected = []
    for folder in ('src', 'tests', 'examples', 'configs'):
        protected.extend(p for p in Path(folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    protected.extend(p for p in Path('.').iterdir() if p.is_file())
    for folder in (BASELINE, Path('results/model_evaluation/phase18b_validation')):
        protected.extend(p for p in folder.rglob('*') if p.is_file())
    trainer.write_json_new(root/'preservation_before.json', {str(p): file_sha256(p) for p in protected})
    dataset_stats = {str(p): [p.stat().st_size, p.stat().st_mtime_ns]
        for folder in ('data/processed/ecg_dataset_v1', 'data/processed/ptbxl_rhythm_v1')
        for p in Path(folder).rglob('*') if p.is_file()}
    trainer.write_json_new(root/'dataset_stats_before.json', dataset_stats)
    reference = json.loads((BASELINE/'result.json').read_text())
    if file_sha256(BASELINE/'best_model.pt') != reference['best_model_sha256']:
        raise ValueError('Baseline checkpoint integrity mismatch')
    index = GuardedIndex(); baseline_plan = json.loads((BASELINE/'selection_manifest.json').read_text())
    if index.manifest_sha256 != baseline_plan['manifest_sha256']:
        raise ValueError('Dataset manifest changed')
    if (len(index.partitions.train), len(index.partitions.validation)) != (15689, 1958):
        raise ValueError('Cohort counts differ')
    resources = system_resources()
    if resources['memory'].get('MemAvailable', 0) < 2**30 or resources['disk_free_bytes'] < 2 * 2**30:
        raise RuntimeError('Insufficient safe RAM/disk reserve before execution')
    times = [json.loads(line) for line in Path('results/model_training/phase18_full/audit_run_001/epoch_times.jsonl').read_text().splitlines()]
    limits = {'maximum_epoch_seconds': 2 * max(row['epoch_seconds'] for row in times),
              'maximum_process_rss_bytes': 2 * 2**30,
              'minimum_available_ram_bytes': 512 * 2**20, 'minimum_disk_bytes': 2**30}
    configs = {}
    for variant in VARIANTS:
        config = configuration(variant)
        with controlled_loss(variant):
            plan = trainer.build_plan(index, config)
        if plan['partitions'] != baseline_plan['partitions']:
            raise ValueError('Training/validation populations or ordering protocol differ')
        configs[variant] = {'training': asdict(config), 'loss_scheme': VARIANTS[variant]}
        trainer.write_json_new(root/f'{variant}_configuration.json', configs[variant])
        trainer.write_json_new(root/f'{variant}_plan.json', plan)
    # Confirm the actual first shuffled batch matches original baseline history.
    cfg = configuration('B_unweighted'); trainer.seed_everything(cfg)
    initial = PTBXLRhythmCNN()
    initial_hash = trainer._hash({k: v.detach().tolist() for k, v in initial.state_dict().items()})
    loader = trainer.make_loader(index, index.partitions.train, 'train', cfg, 1)
    first = next(iter(loader))['ecg_id'].tolist()
    original_first = json.loads((BASELINE/'epoch_001_metrics.json').read_text())['train']['ecg_ids'][:16]
    if first != original_first:
        raise ValueError('Shuffle order differs from Phase 18')
    selected = (*index.partitions.train, *index.partitions.validation)
    for n, pos in enumerate(selected, 1):
        index.load_numpy(pos)
        if n % 1000 == 0:
            print(f'Preflight: {n}/17647 real training+validation NPZ verified', flush=True)
    result = {'status': 'passed', 'baseline_checkpoint': str(BASELINE/'best_model.pt'),
        'baseline_sha256': reference['best_model_sha256'], 'baseline_artifacts_sha256': {
            str(p): file_sha256(p) for p in BASELINE.rglob('*') if p.is_file()},
        'manifest_sha256': index.manifest_sha256, 'resources': resources, 'resource_limits': limits,
        'selection_criterion': SELECTION, 'threshold_grid': [0., .65],
        'training_records': 15689, 'validation_records': 1958, 'verified_npz': len(selected),
        'training_class_counts': training_class_counts(index, index.partitions.train).tolist(),
        'patient_overlap': False, 'test_waveforms_read': 0, 'model_parameter_count': 74116,
        'first_training_batch_matches_baseline': first, 'initial_state_sha256_json': initial_hash,
        'configuration_files': {v: file_sha256(root/f'{v}_configuration.json') for v in VARIANTS},
        'created_utc': datetime.now(timezone.utc).isoformat()}
    trainer.write_json_new(root/'study_plan.json', result)
    return result


def train_variant(root, variant, *, destination=None, resume_from=None, prior_audit=None):
    root = Path(root); configuration(variant)
    plan = json.loads((root/'study_plan.json').read_text())
    if plan['status'] != 'passed':
        raise ValueError('Preflight must pass first')
    config_file = root/f'{variant}_configuration.json'
    if file_sha256(config_file) != plan['configuration_files'][variant]:
        raise ValueError('Prespecified configuration changed')
    settings = json.loads(config_file.read_text())
    config = trainer.TrainingConfig(**settings['training'])
    if asdict(config) != asdict(configuration(variant)):
        raise ValueError('Controlled configuration differs')
    output = root/(destination or variant); audit = root/(output.name+'_audit')
    if output.exists() or audit.exists():
        raise FileExistsError('Resume requires a NEW destination and audit directory')
    audit.mkdir()
    if prior_audit:
        with (audit/'epoch_times.jsonl').open('x') as target:
            target.write((Path(prior_audit)/'epoch_times.jsonl').read_text())
    index = GuardedIndex()
    with controlled_loss(variant):
        current_plan = trainer.build_plan(index, config)
    if current_plan != json.loads((root/f'{variant}_plan.json').read_text()):
        raise ValueError('Code/data/weights changed since preflight')
    trainer.seed_everything(config)
    initial = PTBXLRhythmCNN()
    initial_hash = trainer._hash({k: v.detach().tolist() for k, v in initial.state_dict().items()})
    if initial_hash != plan['initial_state_sha256_json']:
        raise ValueError('Initialization is not identical')
    trainer.write_json_new(audit/'initialization.json', {'initial_state_sha256_json': initial_hash,
        'seed': 42, 'same_as_control': True, 'resume_from': str(resume_from) if resume_from else None})
    previous_signal = signal.getsignal(signal.SIGTERM)
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Preserve completed epochs; resume into a NEW directory')
    signal.signal(signal.SIGTERM, interrupted)
    start = time.perf_counter()
    try:
        with ResourceGuard(audit, plan['resource_limits']) as guard:
            with controlled_loss(variant), epoch_timing(audit, output, index):
                result = trainer.train_model(config, output, index=index, resume_from=resume_from,
                                             allow_full_training=True)
        elapsed = time.perf_counter()-start
        if result['last_completed_checkpoint']:
            _postprocess(output, audit, elapsed)
        trainer.write_json_new(audit/'execution.json', {'variant': variant, 'status': result['status'],
            'duration_seconds': elapsed, 'read_counts_by_fold': index.read_counts,
            'blocked_test_reads': index.blocked_test_reads, 'resource_violation': guard.violation})
        if guard.violation:
            raise RuntimeError(f'Resource limit: {guard.violation}; do not start another run')
        if result['status'] not in ('completed', 'early_stopped'):
            raise RuntimeError('Interrupted: preserve progress and resume; do not start the next variant')
        return result
    finally:
        signal.signal(signal.SIGTERM, previous_signal)


def smoke(root, variant):
    """One real training batch + one small validation batch; full runs reseed fresh."""
    root = Path(root); index = GuardedIndex()
    # Full-training metadata supplies the SAME weights as the future full run;
    # only one batch is executed. Four validation records keep metrics defined.
    config = replace(configuration(variant), epochs=1,
        validation_per_class=1, max_train_batches=1, max_validation_batches=1)
    with controlled_loss(variant):
        result = trainer.train_model(config, root/('smoke_'+variant), index=index,
                                     allow_full_training=True)
    if not result['parameters_changed_this_run'] or not result['checkpoint_reload_identical_logits']:
        raise RuntimeError('Smoke failed')
    trainer.write_json_new(root/('smoke_'+variant+'_audit.json'), {
        'status': 'passed', 'training_records': 16, 'validation_records': 4,
        'optimizer_steps': result['optimizer_steps'], 'test_waveforms_read': 0,
        'blocked_test_reads': index.blocked_test_reads, 'not_used_for_full_model_initialization': True})
    return result
