"""Independent MIT-BIH trainer. No final-test Dataset or preprocessing is available.

Epoch-boundary resume uses a NEW directory and identical source/configuration.
Partial epochs are discarded after interruption. Full execution is approval-gated.
"""
from dataclasses import dataclass, asdict
from itertools import islice
import hashlib
import json
import logging
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from .mitbih_protocol import MITBIHExportIndex, load_split, validate_split, training_class_weights
from .mitbih_torch import MITBIHBeatDataset, collate_mitbih
from .mitbih_model_config import BeatCNNConfig
from .models.mitbih_cnn import MITBIHBeatCNN
from .mitbih_metrics import classification_metrics
from .prepare_dataset import sha256_file
# Read-only reuse of tested persistence/RNG helpers; PTB-XL trainer is unchanged.
from .ptbxl_training import seed_everything, rng_state, restore_rng, save_checkpoint_new, write_json_new

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class MITBIHTrainingConfig:
    version: str = 'mitbih-training-v1'
    epochs: int = 15
    batch_size: int = 32
    learning_rate: float = .001
    seed: int = 42
    device: str = 'cpu'
    cpu_threads: int = 2
    num_workers: int = 0
    train_cache_shards: int = 29
    validation_cache_shards: int = 10
    loss_strategy: str = 'sqrt_balanced'
    patience: int = 4
    min_delta: float = 0.

    def __post_init__(self):
        for name in ('epochs', 'batch_size', 'cpu_threads', 'train_cache_shards', 'validation_cache_shards', 'patience'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if isinstance(self.num_workers, bool) or self.num_workers != 0:
            raise ValueError('This bounded-cache CPU protocol requires num_workers=0')
        if self.device != 'cpu':
            raise ValueError('Only verified CPU execution is enabled in this version')
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2**32:
            raise ValueError('Invalid seed')
        for name in ('learning_rate', 'min_delta'):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or value < 0 or (name == 'learning_rate' and value == 0):
                raise ValueError(f'Invalid {name}')
        if self.loss_strategy not in ('unweighted', 'balanced', 'sqrt_balanced'):
            raise ValueError('Unknown loss strategy')
        if self.version != 'mitbih-training-v1':
            raise ValueError('Unknown configuration version')


def loss_weights(index, split, strategy):
    """Training-only n/(5*n_c), or its square root normalized to mean one."""
    if strategy not in ('unweighted', 'balanced', 'sqrt_balanced'):
        raise ValueError('Unknown loss strategy')
    validate_split(index, split)
    if strategy == 'unweighted':
        return None
    weights = training_class_weights(index, split, split['groups']['train'])
    if strategy == 'sqrt_balanced':
        weights = np.sqrt(weights.astype(np.float64))
        weights /= weights.mean()
    return np.asarray(weights, dtype=np.float32)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def build_plan(index, split, config):
    summary = validate_split(index, split)
    weights = loss_weights(index, split, config.loss_strategy)
    sources = ('mitbih_training.py', 'mitbih_metrics.py', 'mitbih_protocol.py', 'mitbih_torch.py',
               'mitbih_model_config.py', 'models/mitbih_cnn.py', 'ptbxl_metrics.py', 'ptbxl_training.py')
    plan = dict(schema='mitbih-training-plan-v1', configuration=asdict(config),
        manifest_sha256=index.manifest_sha256, split_sha256=fingerprint(split),
        split=split, summary=summary, model_configuration=BeatCNNConfig().to_dict(),
        class_names=list('NSVFQ'), parameter_count=BeatCNNConfig().expected_parameter_count(),
        runtime_versions=dict(torch=str(torch.__version__), numpy=np.__version__),
        class_weights=None if weights is None else weights.tolist(),
        class_weight_source='training_groups_only', final_test_used=False,
        code_sha256={name:sha256_file(Path(__file__).parent/name) for name in sources})
    return json.loads(json.dumps(plan))


def make_loader(dataset, config, epoch):
    if dataset.partition not in ('train', 'validation'):
        raise PermissionError('Final test is unavailable in the MIT-BIH trainer')
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=dataset.partition == 'train',
        num_workers=0, generator=torch.Generator().manual_seed(config.seed+epoch),
        collate_fn=collate_mitbih, drop_last=False)


def epoch_pass(model, loader, split, *, partition, weights=None, optimizer=None, batch_cap=None):
    """Exact weighted CE numerator/denominator; eval forward-only CPU timing.

    Full trainer passes no caps and checks exact source coverage. Caps are solely
    for controlled synthetic tests. Training metrics use online pre-update logits.
    """
    if partition not in ('train', 'validation') or (optimizer is not None) != (partition == 'train'):
        raise PermissionError('Optimizer only for training; final test forbidden')
    if batch_cap is not None and (isinstance(batch_cap, bool) or not isinstance(batch_cap, int) or batch_cap < 1):
        raise ValueError('Invalid batch cap')
    model.train(partition == 'train')
    criterion = torch.nn.CrossEntropyLoss(weight=weights, reduction='sum')
    numerator = denominator = forward_seconds = loading_seconds = 0.
    labels = []; probabilities = []; groups = []; source_indices = []
    start = time.perf_counter(); iterator = iter(loader); batches = 0
    context = torch.enable_grad() if optimizer is not None else torch.inference_mode()
    with context:
        for _ in range(len(loader) if batch_cap is None else min(len(loader), batch_cap)):
            tick = time.perf_counter(); batch = next(iterator); loading_seconds += time.perf_counter()-tick
            if not set(batch['patient_group']) <= set(split['groups'][partition]):
                raise RuntimeError('Patient group leakage in batch')
            x, y = batch['X'], batch['y']
            if x.dtype != torch.float32 or tuple(x.shape[1:]) != (2, 252) or y.dtype != torch.long or y.shape != (len(x),):
                raise ValueError('Invalid heartbeat tensors')
            if not bool(((y >= 0) & (y < 5)).all()):
                raise ValueError('Invalid beat labels')
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            tick = time.perf_counter(); logits = model(x); forward_seconds += time.perf_counter()-tick
            if logits.shape != (len(y), 5) or not bool(torch.isfinite(logits).all()):
                raise FloatingPointError('Invalid logits')
            loss_sum = criterion(logits, y)
            divisor = weights[y].sum() if weights is not None else torch.tensor(float(len(y)))
            loss = loss_sum/divisor
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Nonfinite loss')
            if optimizer is not None:
                loss.backward()
                if not all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.requires_grad):
                    raise FloatingPointError('Invalid gradients')
                optimizer.step()
                if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
                    raise FloatingPointError('Invalid updated weights')
            numerator += float(loss_sum.detach()); denominator += float(divisor)
            labels.extend(y.tolist()); probabilities.extend(torch.softmax(logits.detach(), 1).tolist())
            groups.extend(batch['patient_group']); source_indices.extend(batch['global_source_index'].tolist())
            batches += 1
            if batches == 1 or batches % 100 == 0:
                LOG.info('%s batch %s/%s', partition, batches, len(loader))
    if not labels or len(set(source_indices)) != len(source_indices):
        raise ValueError('Empty or duplicate source samples')
    if batch_cap is None:
        expected = {e.global_start+i for e in loader.dataset.shards for i in range(e.count)}
        if set(source_indices) != expected:
            raise RuntimeError('Incomplete partition coverage')
    metrics = classification_metrics(np.asarray(labels, dtype=np.int64), np.asarray(probabilities))
    per_group = {}
    for g in sorted(set(groups)):
        mask = np.asarray(groups) == g
        per_group[g] = classification_metrics(np.asarray(labels, dtype=np.int64)[mask], np.asarray(probabilities)[mask])
    metrics.update(loss=numerator/denominator, loss_numerator=numerator, loss_denominator=denominator,
        batches=batches, forward_seconds=forward_seconds, loading_seconds=loading_seconds,
        total_seconds=time.perf_counter()-start,
        inference_ms_per_beat=1000*forward_seconds/len(labels) if optimizer is None else None,
        per_group=per_group, complete_partition=batch_cap is None,
        metric_scope='online pre-update training' if optimizer is not None else 'epoch-end full validation')
    return metrics


def load_checkpoint(path, plan):
    """Trusted local, weights-only checkpoint with sibling SHA256 and plan identity."""
    path = Path(path)
    if sha256_file(path) != path.with_suffix('.sha256').read_text().strip():
        raise ValueError('Checkpoint SHA256 mismatch')
    c = torch.load(path, map_location='cpu', weights_only=True)
    if c.get('schema') != 'mitbih-training-checkpoint-v1' or c.get('plan') != plan or c.get('plan_sha256') != fingerprint(plan):
        raise ValueError('Checkpoint plan/config/source identity differs')
    if c['early_stopped'] or c['completed_epoch'] >= plan['configuration']['epochs']:
        raise ValueError('Run is already complete or early-stopped')
    if len(c['history']) != c['completed_epoch'] or c['best_epoch'] not in range(1, c['completed_epoch']+1):
        raise ValueError('Invalid checkpoint history')
    for state in (c['model_state_dict'], c['best_model_state_dict']):
        if not all(bool(torch.isfinite(t).all()) for t in state.values()):
            raise ValueError('Nonfinite checkpoint weights')
    for state in c['optimizer_state_dict']['state'].values():
        for value in state.values():
            if isinstance(value, torch.Tensor) and not bool(torch.isfinite(value).all()):
                raise ValueError('Nonfinite checkpoint optimizer state')
    return c


def train_model(config, output_dir, *, index=None, split=None, resume_from=None,
                allow_full_training=False, epoch_budget=None):
    """Full training requires explicit approval. Never called by readiness benchmark.

    Immutable epoch files are resumable only at completed epoch boundaries. New
    output directory on resume; partial epoch updates are discarded, no silent skip.
    """
    if allow_full_training is not True:
        raise PermissionError('Full MIT-BIH training requires separate explicit approval')
    if epoch_budget is not None and (isinstance(epoch_budget, bool) or not isinstance(epoch_budget, int) or epoch_budget < 1):
        raise ValueError('Invalid epoch budget')
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError('Use a NEW experiment directory, including for resume')
    index = MITBIHExportIndex() if index is None else index
    split = load_split(index) if split is None else split
    if len(index.groups) == 47:
        from .mitbih_readiness import verify_frozen_protocol
        verify_frozen_protocol(index, split)
    plan = build_plan(index, split, config)
    seed_everything(config)
    model = MITBIHBeatCNN(); optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    weights = None if plan['class_weights'] is None else torch.tensor(plan['class_weights'], dtype=torch.float32)
    history = []; best_score = -1.; best_epoch = None; best_state = None; stale = 0; stop_score = -1.; first = 1
    if resume_from is not None:
        c = load_checkpoint(resume_from, plan)
        model.load_state_dict(c['model_state_dict']); optimizer.load_state_dict(c['optimizer_state_dict'])
        history = c['history']; best_score = c['best_score']; best_epoch = c['best_epoch']; best_state = c['best_model_state_dict']
        stale = c['stale_epochs']; stop_score = c['stopping_score']; first = c['completed_epoch']+1
        restore_rng(c['rng_state'])
    train = MITBIHBeatDataset(index, split, partition='train', max_cached_shards=config.train_cache_shards)
    validation = MITBIHBeatDataset(index, split, partition='validation', max_cached_shards=config.validation_cache_shards)
    output.mkdir(parents=True, exist_ok=False)
    write_json_new(output/'training_plan.json', plan)
    write_json_new(output/'configuration.json', asdict(config))
    handler = logging.FileHandler(output/'training.log', mode='x')
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    LOG.addHandler(handler); LOG.setLevel(logging.INFO)
    status = 'completed'; last = None; executed = 0
    try:
        for epoch in range(first, config.epochs+1):
            if epoch_budget is not None and executed >= epoch_budget:
                status = 'invocation_budget'; break
            started = time.perf_counter()
            tr = epoch_pass(model, make_loader(train, config, epoch), split, partition='train', weights=weights, optimizer=optimizer)
            va = epoch_pass(model, make_loader(validation, config, epoch), split, partition='validation', weights=weights)
            score = va['macro_f1']
            if score is None:
                raise ValueError('Full validation Macro F1 undefined')
            row = dict(epoch=epoch, train=tr, validation=va, seconds=time.perf_counter()-started)
            history.append(row)
            if score > best_score:
                best_score = score; best_epoch = epoch
                best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            if score > stop_score+config.min_delta:
                stop_score = score; stale = 0
            else:
                stale += 1
            stopped = stale >= config.patience
            payload = dict(schema='mitbih-training-checkpoint-v1', plan=plan, plan_sha256=fingerprint(plan),
                completed_epoch=epoch, history=history, model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(), best_model_state_dict=best_state,
                best_score=best_score, best_epoch=best_epoch, stale_epochs=stale, stopping_score=stop_score,
                early_stopped=stopped, rng_state=rng_state())
            last = output/f'epoch_{epoch:03d}.pt'
            save_checkpoint_new(last, payload)
            with last.with_suffix('.sha256').open('x') as f:
                f.write(sha256_file(last)+'\n')
            write_json_new(output/f'epoch_{epoch:03d}_metrics.json', row)
            executed += 1
            LOG.info('epoch %s train loss %.6f validation loss %.6f Macro F1 %.6f', epoch, tr['loss'], va['loss'], score)
            if stopped:
                status = 'early_stopped'; break
    except KeyboardInterrupt:
        status = 'interrupted_partial_epoch_discarded'
    except Exception as exc:
        write_json_new(output/'failure.json', dict(error=f'{type(exc).__name__}: {exc}', last_checkpoint=str(last) if last else None))
        raise
    finally:
        LOG.removeHandler(handler); handler.close()
    if best_state is not None:
        target = output/'best_model.pt'
        save_checkpoint_new(target, dict(schema='mitbih-best-model-v1', model_state_dict=best_state,
            model_configuration=BeatCNNConfig().to_dict(), class_names=list('NSVFQ'),
            best_epoch=best_epoch, best_validation_macro_f1=best_score, plan_sha256=fingerprint(plan)))
        with target.with_suffix('.sha256').open('x') as f:
            f.write(sha256_file(target)+'\n')
    result = dict(status=status, last_checkpoint=str(last) if last else str(resume_from) if resume_from else None,
        completed_epochs=len(history), best_epoch=best_epoch, best_validation_macro_f1=best_score if history else None,
        executed_this_run=executed, final_test_used=False)
    write_json_new(output/'result.json', result)
    return result
