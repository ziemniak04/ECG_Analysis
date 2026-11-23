"""Phase-22 orchestration and frozen-run reporting, without final-test inference."""
from dataclasses import asdict
import csv
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np
import torch
from .mitbih_protocol import MITBIHExportIndex, load_split
from .mitbih_training import MITBIHTrainingConfig, build_plan, train_model, loss_weights, fingerprint
from .mitbih_readiness import verify_frozen_protocol, resources
from .models.mitbih_cnn import MITBIHBeatCNN
from .mitbih_torch import MITBIHBeatDataset, collate_mitbih
from .prepare_dataset import sha256_file
from .ptbxl_training import write_json_new

APPROVED = dict(version='mitbih-training-v1', epochs=15, batch_size=32, learning_rate=.001,
    seed=42, device='cpu', cpu_threads=2, num_workers=0, train_cache_shards=29,
    validation_cache_shards=10, loss_strategy='sqrt_balanced', patience=4, min_delta=0.)


def verify_config(config):
    if asdict(config) != APPROVED:
        raise ValueError('Configuration differs from the explicitly approved Phase-22 settings')


def snapshot(audit):
    """Protect earlier sources, reports, artifacts, datasets without decoding test NPZ."""
    paths = [p for p in Path('.').iterdir() if p.is_file()]
    for name in ('src', 'examples', 'tests', 'configs', 'results/model_training',
                 'results/model_evaluation', 'results/model_setup'):
        paths.extend(p for p in Path(name).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    paths = [p for p in paths if not p.is_relative_to(audit)]
    write_json_new(audit/'preservation_before.json', {str(p):sha256_file(p) for p in paths})
    write_json_new(audit/'dataset_stats_before.json', {str(p):[p.stat().st_size,p.stat().st_mtime_ns]
        for root in ('data/processed/ecg_dataset_v1','data/processed/ptbxl_rhythm_v1')
        for p in Path(root).rglob('*') if p.is_file()})


def preflight(config, audit, output):
    """Validate every TRAIN/VALIDATION shard; final-test files are stat-only."""
    verify_config(config); audit = Path(audit); output = Path(output)
    if audit.exists() or output.exists():
        raise FileExistsError('New preflight and experiment directories are required')
    if Path('PHASE22_MITBIH_FULL_TRAINING_REPORT.md').exists():
        raise FileExistsError('Phase-22 report already exists; inspect completed work first')
    index = MITBIHExportIndex(); split = load_split(index)
    summary = verify_frozen_protocol(index, split)
    capacity = resources()
    if capacity['MemAvailable_bytes'] < 2*2**30 or capacity['disk_free_bytes'] < 64*2**20:
        raise RuntimeError('Insufficient current RAM/disk headroom; do not start training')
    model = MITBIHBeatCNN()
    if model.parameter_count() != 17797:
        raise ValueError('Architecture differs from the approved model')
    audit.mkdir(parents=True, exist_ok=False); snapshot(audit)
    checks = {}
    for part in ('train', 'validation'):
        count = np.zeros(5, dtype=np.int64)
        for rid in summary[part]['record_ids']:
            arrays = index.load_shard(rid, split, part)
            count += np.bincount(arrays['y'], minlength=5)
            checks[rid] = index.by_record[rid].sha256
        if count.tolist() != [summary[part]['class_counts'][c] for c in 'NSVFQ']:
            raise ValueError('Actual TRAIN/VALIDATION labels differ from approved counts')
    # Check the optional weights independently from TRAIN counts only.
    n = np.asarray([summary['train']['class_counts'][c] for c in 'NSVFQ'], dtype=float)
    manual = np.sqrt(n.sum()/(5*n)); manual /= manual.mean()
    np.testing.assert_allclose(loss_weights(index, split, config.loss_strategy), manual, rtol=1e-6)
    plan = build_plan(index, split, config)
    write_json_new(audit/'preflight.json', dict(status='passed', configuration=asdict(config),
        configuration_sha256=sha256_file('PHASE21_MITBIH_TRAINING_CONFIG.json'), summary=summary,
        resources=capacity, parameter_count=17797, class_names=list('NSVFQ'),
        actual_shard_sha256_and_labels_verified=checks,
        class_weights=manual.tolist(), final_test_npz_opened=False, output_dir=str(output),
        source_manifest_sha256=index.manifest_sha256, training_plan_sha256=fingerprint(plan)))
    write_json_new(audit/'reproducibility.json', dict(python=sys.version, platform=platform.platform(),
        torch=str(torch.__version__), numpy=np.__version__, seed=config.seed,
        cpu_threads=config.cpu_threads, deterministic_algorithms=True,
        dependencies={d.metadata['Name']:d.version for d in importlib.metadata.distributions() if d.metadata['Name']},
        approval='User explicitly approved Phase-21 Checkpoint B / Phase-22 first full run',
        no_preprocessing_repeated=True, test_evaluation_authorized=False))
    return index, split


def run_approved(config, audit, output):
    """Exactly one full experiment. No automatic restart or second configuration."""
    index, split = preflight(config, audit, output)
    print(json.dumps({'preflight':'passed', 'output':str(output), 'configuration':asdict(config)},indent=2),flush=True)
    started = time.perf_counter()
    result = train_model(config, output, index=index, split=split, allow_full_training=True)
    elapsed = time.perf_counter()-started
    allowed = set(split['summary']['train']['record_ids']+split['summary']['validation']['record_ids'])
    if not set(index.read_records) <= allowed:
        raise RuntimeError('Unexpected final-test shard read')
    write_json_new(Path(output)/'execution.json', dict(wall_seconds=elapsed, resources_after=resources(),
        read_records_including_preflight=index.read_records, final_test_read=False, result=result,
        timing_scope='Entire train_model call, including loading, metrics and checkpointing; excludes preflight'))
    return result


def validated_history(output):
    """Read completed metrics/checkpoints only; no dataset or model forward."""
    output = Path(output); result = json.loads((output/'result.json').read_text())
    plan = json.loads((output/'training_plan.json').read_text())
    history = []
    for epoch in range(1,result['completed_epochs']+1):
        row = json.loads((output/f'epoch_{epoch:03d}_metrics.json').read_text())
        if row['epoch'] != epoch:
            raise ValueError('Nonsequential epochs')
        for part in ('train','validation'):
            m = row[part]; cm = np.asarray(m['confusion_matrix'])
            expected = plan['summary'][part]['class_counts']
            if (m['beat_count'] != sum(expected.values()) or not m['complete_partition']
                    or cm.shape != (5,5) or cm.sum(axis=1).tolist() != [expected[c] for c in 'NSVFQ']
                    or set(m['per_group']) != set(plan['split']['groups'][part])):
                raise ValueError('Epoch population/counts disagree')
            np.testing.assert_allclose(m['accuracy'], np.trace(cm)/cm.sum(), rtol=0, atol=1e-12)
            np.testing.assert_allclose(m['macro_f1'], np.mean([r['f1'] for r in m['per_class']]), rtol=0, atol=1e-12)
            np.testing.assert_allclose(m['macro_auprc'], np.mean([r['auprc'] for r in m['per_class']]), rtol=0, atol=1e-12)
            np.testing.assert_allclose(sum(np.asarray(g['confusion_matrix']) for g in m['per_group'].values()), cm)
            if m['absent_true_classes'] or not np.isfinite(m['loss']):
                raise ValueError('Incomplete classes or nonfinite epoch loss')
        cp = output/f'epoch_{epoch:03d}.pt'
        if sha256_file(cp) != cp.with_suffix('.sha256').read_text().strip():
            raise ValueError('Epoch checksum mismatch')
        history.append(row)
    best = max(history,key=lambda r:r['validation']['macro_f1'])
    if best['epoch'] != result['best_epoch'] or best['validation']['macro_f1'] != result['best_validation_macro_f1']:
        raise ValueError('Best selection disagrees with history')
    return result, plan, history, best


def csv_new(path, rows):
    if not rows: raise ValueError('Nonempty rows required')
    with Path(path).open('x',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def summarize_run(output):
    """Create plots/tables, verify best reload on TRAIN ONLY, preserve all inputs."""
    output = Path(output); result, plan, history, best = validated_history(output)
    target = output/'analysis'
    if target.exists(): raise FileExistsError('Analysis already exists')
    target.mkdir()
    write_json_new(target/'training_history.json', history)
    csv_new(target/'training_history.csv', [dict(epoch=r['epoch'],epoch_seconds=r['seconds'],
        train_loss=r['train']['loss'],validation_loss=r['validation']['loss'],
        train_accuracy=r['train']['accuracy'],validation_accuracy=r['validation']['accuracy'],
        validation_macro_f1=r['validation']['macro_f1'],validation_macro_auprc=r['validation']['macro_auprc']) for r in history])
    write_json_new(target/'best_validation_metrics.json', best['validation'])
    csv_new(target/'per_class_validation_metrics.csv', best['validation']['per_class'])
    group_rows = [dict(patient_group=g, **row) for g,m in best['validation']['per_group'].items() for row in m['per_class']]
    csv_new(target/'patient_group_class_metrics.csv',group_rows)
    write_json_new(target/'patient_group_validation_metrics.json', best['validation']['per_group'])
    checkpoint = output/'best_model.pt'
    if sha256_file(checkpoint) != checkpoint.with_suffix('.sha256').read_text().strip():
        raise ValueError('Best checksum mismatch')
    saved = torch.load(checkpoint,map_location='cpu',weights_only=True)
    last = torch.load(result['last_checkpoint'],map_location='cpu',weights_only=True)
    if (saved['class_names'] != list('NSVFQ') or saved['plan_sha256'] != fingerprint(plan)
            or last['plan'] != plan or saved['best_epoch'] != best['epoch']):
        raise ValueError('Best checkpoint identity mismatch')
    reference = MITBIHBeatCNN();reference.load_state_dict(last['best_model_state_dict']);reference.eval()
    reloaded = MITBIHBeatCNN();reloaded.load_state_dict(saved['model_state_dict']);reloaded.eval()
    for k,v in reference.state_dict().items():
        torch.testing.assert_close(v,reloaded.state_dict()[k],rtol=0,atol=0)
    torch.set_num_threads(2)
    index = MITBIHExportIndex(); split = load_split(index); verify_frozen_protocol(index,split)
    dataset = MITBIHBeatDataset(index,split,partition='train',max_cached_shards=1)
    batch = collate_mitbih([dataset[i] for i in range(8)])
    with torch.inference_mode():
        a,b = reference(batch['X']),reloaded(batch['X'])
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    if not torch.isfinite(a).all():raise ValueError('Nonfinite reload logits')
    write_json_new(target/'checkpoint_reload_check.json', dict(status='passed', identical_state=True,
        identical_logits=True, batch_shape=list(batch['X'].shape), output_shape=list(a.shape),
        partition='train', records=index.read_records, parameters_unchanged=True,
        best_sha256=sha256_file(checkpoint), final_test_read=False))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    epochs = [r['epoch'] for r in history]
    for filename,ylabel,series in [
        ('loss_curves.png','Weighted cross-entropy', [('Train (online)',[r['train']['loss'] for r in history]),('Validation',[r['validation']['loss'] for r in history])]),
        ('validation_macro_f1.png','Validation Macro F1',[('Full validation',[r['validation']['macro_f1'] for r in history])]),
        ('validation_macro_auprc.png','Validation Macro AP',[('Full validation',[r['validation']['macro_auprc'] for r in history])])]:
        fig,ax = plt.subplots(figsize=(8,4.5),layout='constrained')
        for name,values in series:ax.plot(epochs,values,'o-',label=name)
        ax.axvline(best['epoch'],color='grey',linestyle='--',label='Selected epoch')
        ax.set(xlabel='Completed epoch',ylabel=ylabel,title='MIT-BIH — patient-separated training / validation')
        ax.set_xticks(epochs);ax.grid(alpha=.25);ax.legend()
        with (target/filename).open('xb') as f:fig.savefig(f,dpi=300,format='png')
        plt.close(fig)
    cm = np.asarray(best['validation']['confusion_matrix'])
    csv_new(target/'confusion_matrix.csv',[dict(true_class=c,**dict(zip('NSVFQ',map(int,cm[i])))) for i,c in enumerate('NSVFQ')])
    for norm in (False,True):
        values = cm/cm.sum(axis=1,keepdims=True) if norm else cm
        fig,ax = plt.subplots(figsize=(6.4,5.5),layout='constrained')
        im=ax.imshow(values,cmap='Blues',vmin=0,vmax=1 if norm else None);fig.colorbar(im,ax=ax)
        for i in range(5):
            for j in range(5):
                label=f'{values[i,j]:.1%}' if norm else str(cm[i,j])
                ax.text(j,i,label,ha='center',va='center',color='white' if values[i,j]>values.max()/2 else 'black')
        ax.set_xticks(range(5),list('NSVFQ'));ax.set_yticks(range(5),list('NSVFQ'))
        ax.set(xlabel='Predicted beat group',ylabel='True beat group',title=f'MIT-BIH validation — selected epoch {best["epoch"]}')
        with (target/('validation_confusion_normalized.png' if norm else 'validation_confusion_matrix.png')).open('xb') as f:
            fig.savefig(f,dpi=300,format='png')
        plt.close(fig)
    return result, best
