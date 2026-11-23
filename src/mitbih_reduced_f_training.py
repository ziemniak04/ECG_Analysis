"""Phase25B controlled plan; orchestration body identical to frozen Phase24B.

Only F weight is halved after the original TRAIN-only sqrt-balanced calculation.
No renormalization. Epochs, data, initialization, optimizer and persistence are
unchanged. The separate module preserves every historical source file.
"""
from dataclasses import asdict
from pathlib import Path
import json,logging,time
import numpy as np
import torch
from . import mitbih_cnn_rr_training as reference
from .mitbih_cnn_rr_training import (MITBIHTrainingConfig,MITBIHExportIndex,load_split,
    MITBIHBeatCNNRR,MITBIHRRDataset,DEFAULT_RR_ROOT,require_guard,seed_everything,
    restore_rng,rng_state,make_loader,epoch_pass,load_checkpoint,save_checkpoint_new,
    write_json_new,sha256_file,fingerprint)
LOG=logging.getLogger(__name__)


def build_plan(index,split,config,rr_root=DEFAULT_RR_ROOT):
    require_guard()
    if config.loss_strategy!='sqrt_balanced':raise ValueError('Phase25B permits sqrt-balanced with F*0.5 only')
    plan=reference.build_plan(index,split,config,rr_root)
    original=list(plan['class_weights']);weights=list(original);weights[3]*=.5
    plan['original_class_weights']=original
    plan['class_weights']=weights
    plan['loss_weight_adjustment']=dict(version='phase25b-F-half-v1',class_name='F',class_index=3,multiplier=.5,renormalized=False,other_weights_unchanged=True)
    plan['code_sha256']['mitbih_reduced_f_training.py']=sha256_file(Path(__file__))
    return plan


def train_model(config, output_dir, *, index=None, split=None, resume_from=None,
                allow_full_training=False, epoch_budget=None,rr_root=DEFAULT_RR_ROOT):
    """Full training requires explicit approval. Never called by readiness benchmark.

    Immutable epoch files are resumable only at completed epoch boundaries. New
    output directory on resume; partial epoch updates are discarded, no silent skip.
    """
    require_guard()
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
    plan = build_plan(index, split, config,rr_root)
    seed_everything(config)
    model = MITBIHBeatCNNRR(); optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    weights = None if plan['class_weights'] is None else torch.tensor(plan['class_weights'], dtype=torch.float32)
    history = []; best_score = -1.; best_epoch = None; best_state = None; stale = 0; stop_score = -1.; first = 1
    if resume_from is not None:
        c = load_checkpoint(resume_from, plan)
        model.load_state_dict(c['model_state_dict']); optimizer.load_state_dict(c['optimizer_state_dict'])
        history = c['history']; best_score = c['best_score']; best_epoch = c['best_epoch']; best_state = c['best_model_state_dict']
        stale = c['stale_epochs']; stop_score = c['stopping_score']; first = c['completed_epoch']+1
        restore_rng(c['rng_state'])
    train = MITBIHRRDataset(rr_root,index=index,split=split,partition='train', max_cached_shards=config.train_cache_shards)
    validation = MITBIHRRDataset(rr_root,index=index,split=split,partition='validation', max_cached_shards=config.validation_cache_shards)
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
            payload = dict(schema='mitbih-cnn-rr-training-checkpoint-v1', plan=plan, plan_sha256=fingerprint(plan),
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
        save_checkpoint_new(target, dict(schema='mitbih-cnn-rr-best-model-v1', model_state_dict=best_state,
            model_configuration=model.configuration(),preprocessing_configuration=dict(ECG=plan['ECG_preprocessing_by_record'],RR=plan['rr_configuration'],RR_transform=plan['rr_training_transform']),rr_manifest_sha256=plan['rr_manifest_sha256'], class_names=list('NSVFQ'),
            best_epoch=best_epoch, best_validation_macro_f1=best_score, plan_sha256=fingerprint(plan)))
        with target.with_suffix('.sha256').open('x') as f:
            f.write(sha256_file(target)+'\n')
    result = dict(status=status, last_checkpoint=str(last) if last else str(resume_from) if resume_from else None,
        completed_epochs=len(history), best_epoch=best_epoch, best_validation_macro_f1=best_score if history else None,
        executed_this_run=executed, final_test_used=False)
    write_json_new(output/'result.json', result)
    return result
