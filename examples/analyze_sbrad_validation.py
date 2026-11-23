"""Evaluate a frozen Phase-18 model on fold 9 and study SBRAD abstention thresholds."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch

from src.models.ptbxl_cnn import PTBXLRhythmCNN
from src.ptbxl_decisions import (ValidationOnlyIndex, frozen_validation_predictions,
    sbrad_decisions, threshold_study, recommend_threshold, THRESHOLDS, SELECTION)
from src.ptbxl_metrics import CLASS_NAMES, classification_metrics
from src.ptbxl_model_config import RhythmCNNConfig
from src.ptbxl_protocol import file_sha256


def write_json(path, value):
    with path.open('x') as target:
        json.dump(value, target, indent=2, allow_nan=False)


def write_csv(path, rows):
    with path.open('x', newline='') as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def analyze(output, checkpoint):
    output, checkpoint = Path(output), Path(checkpoint)
    if output.exists():
        raise FileExistsError('Use a new directory; existing Phase-18B outputs are immutable')
    checkpoint_sha = file_sha256(checkpoint)
    phase18 = checkpoint.parent
    previous_result = json.loads((phase18 / 'result.json').read_text())
    if checkpoint_sha != previous_result['best_model_sha256']:
        raise ValueError('Checkpoint differs from verified Phase-18 model')
    protected_paths = [p for folder in ('src', 'configs') for p in Path(folder).rglob('*.py' if folder == 'src' else '*.json')]
    protected_paths += [p for p in Path('.').iterdir() if p.is_file()]
    protected_paths += [p for p in Path('results/model_training').rglob('*') if p.is_file()]
    protected = {str(p): file_sha256(p) for p in protected_paths}
    dataset_stats = {str(p): [p.stat().st_size, p.stat().st_mtime_ns]
                     for root in ('data/processed/ecg_dataset_v1', 'data/processed/ptbxl_rhythm_v1')
                     for p in Path(root).rglob('*') if p.is_file()}
    output.mkdir(parents=True, exist_ok=False)
    settings = {'checkpoint': str(checkpoint), 'checkpoint_sha256': checkpoint_sha,
                'device': 'cpu', 'cpu_threads': 2, 'batch_size': 16, 'seed': 42,
                'waveform_fold': 9, 'threshold_grid': list(THRESHOLDS),
                'selection_criterion_before_study': SELECTION,
                'probability_calibration_fitted': False, 'architecture_changed': False,
                'preprocessing_changed': False, 'created_utc': datetime.now(timezone.utc).isoformat()}
    write_json(output / 'study_configuration.json', settings)
    write_json(output / 'preservation_before.json', protected)
    write_json(output / 'dataset_stats_before.json', dataset_stats)
    torch.set_num_threads(2); torch.manual_seed(42); torch.use_deterministic_algorithms(True)
    index = ValidationOnlyIndex()
    if len(index.partitions.validation) != 1958:
        raise ValueError('Full validation count differs')
    frozen = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if frozen['schema'] != 'ptbxl-best-model-v1' or frozen['classes'] != list(CLASS_NAMES):
        raise ValueError('Checkpoint schema or label mapping differs')
    if frozen['manifest_sha256'] != index.manifest_sha256:
        raise ValueError('Checkpoint/export identity differs')
    model = PTBXLRhythmCNN(RhythmCNNConfig(**frozen['model_configuration']))
    assert model.parameter_count() == 74116
    model.load_state_dict(frozen['model_state_dict'], strict=True)
    start = time.perf_counter()
    data = frozen_validation_predictions(model, index)
    elapsed = time.perf_counter() - start
    y, prob = data['labels'], data['probabilities']
    baseline = classification_metrics(y, prob)
    old = json.loads((phase18 / 'best_validation_metrics.json').read_text())
    if (data['ecg_id'].tolist() != old['ecg_ids'] or y.tolist() != old['true_labels']
            or baseline['confusion_matrix'] != old['confusion_matrix']
            or not np.allclose(prob, old['probabilities'], atol=1e-7, rtol=1e-6)):
        raise ValueError('Fresh inference does not reproduce Phase-18 validation predictions')
    write_json(output / 'baseline_metrics.json', baseline)
    with (output / 'validation_predictions.npz').open('xb') as target:
        np.savez_compressed(target, **data)
    rows = threshold_study(y, prob)
    selection = recommend_threshold(rows)
    write_json(output / 'threshold_metrics.json', rows)
    write_json(output / 'decision_rule.json', {
        **selection, 'class_names': list(CLASS_NAMES), 'sbrad_index': 3,
        'abstain_index': -1, 'only_reject_sbrad_argmax_winners': True,
        'accept_equality': True, 'second_choice_fallback': False,
        'validation_selected_not_medically_validated': True,
        'checkpoint_sha256': checkpoint_sha, 'probability_calibration_fitted': False})
    table = []
    for row in rows:
        s = row['per_class'][3]
        table.append({'threshold': row['threshold'], 'rule': 'argmax' if row['threshold'] == 0 else 'SBRAD gate + ABSTAIN',
            'sbrad_tp': s['tp'], 'sbrad_fp': s['fp'], 'sbrad_fn': s['fn'], 'sbrad_tn': s['tn'],
            'sbrad_precision': s['precision'], 'sbrad_recall_full_cohort': s['recall'],
            'sbrad_specificity': s['specificity'], 'sbrad_f1_full_cohort': s['f1'],
            'abstained': row['abstained'], 'accepted': row['accepted'],
            'overall_abstention_rate': row['abstention_rate'], 'coverage': row['coverage'],
            **{f"{r['class']}_abstention_rate": r['abstention_rate'] for r in row['per_class']},
            **{k: row[k] for k in ('accepted_accuracy_coverage_conditional', 'accepted_macro_f1_coverage_conditional',
                                    'full_cohort_macro_f1_abstentions_as_misses')},
            'eligible': row['threshold'] in selection['eligible_thresholds'],
            'recommended': row['threshold'] == selection['selected_threshold']})
    write_csv(output / 'SBRAD_THRESHOLD_COMPARISON.csv', table)
    predicted = prob.argmax(axis=1)
    masks = {'correct_SBRAD': (y == 3) & (predicted == 3),
             'false_SBRAD': (y != 3) & (predicted == 3),
             'missed_SBRAD': (y == 3) & (predicted != 3)}
    summaries = []
    for group, mask in masks.items():
        scores = prob[mask, 3]
        summaries.append({'group': group, 'count': len(scores),
            'mean': float(scores.mean()) if len(scores) else None,
            **{name: float(np.quantile(scores, q)) if len(scores) else None
               for name, q in [('minimum', 0), ('q25', .25), ('median', .5), ('q75', .75), ('maximum', 1)]}})
    write_csv(output / 'sbrad_confidence_summary.csv', summaries)
    # Rank discrimination among existing SBRAD alerts, not probability calibration.
    from scipy.stats import rankdata
    positive = prob[masks['correct_SBRAD'], 3]; negative = prob[masks['false_SBRAD'], 3]
    if len(positive) and len(negative):
        ranks = rankdata(np.r_[positive, negative])[:len(positive)]
        ranking_auc = float((ranks.sum() - len(positive)*(len(positive)+1)/2) / (len(positive)*len(negative)))
    else:
        ranking_auc = None
    source = index.source_metadata.records.set_index('ecg_id')
    selected = next(r for r in rows if r['threshold'] == selection['selected_threshold'])
    decision = sbrad_decisions(prob, selection['selected_threshold'])
    prediction_rows = []
    for i, eid in enumerate(data['ecg_id']):
        prediction_rows.append({'ecg_id': int(eid), 'patient_id': int(data['patient_id'][i]), 'strat_fold': 9,
            'true_class': CLASS_NAMES[y[i]], 'argmax_class': CLASS_NAMES[predicted[i]],
            'selected_rule_class': 'ABSTAIN' if decision[i] == -1 else CLASS_NAMES[decision[i]],
            **{f'p_{name}': float(prob[i, j]) for j, name in enumerate(CLASS_NAMES)},
            'original_scp_codes': str(source.loc[int(eid), 'scp_codes'])})
    write_csv(output / 'validation_predictions.csv', prediction_rows)
    write_csv(output / 'argmax_errors.csv', [r for r in prediction_rows if r['true_class'] != r['argmax_class']])
    write_csv(output / 'sbrad_false_positives.csv', [r for r in prediction_rows if r['argmax_class'] == 'SBRAD' and r['true_class'] != 'SBRAD'])
    write_csv(output / 'per_class_threshold_metrics.csv', [
        {'threshold': r['threshold'], **c} for r in rows for c in r['per_class']])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def save(fig, name):
        with (output / name).open('xb') as target: fig.savefig(target, format='png', dpi=300)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
    valid = [r for r in table if r['sbrad_precision'] is not None]
    axes[0].plot([r['sbrad_recall_full_cohort'] for r in valid], [r['sbrad_precision'] for r in valid], 'o-')
    for r in valid:
        if r['threshold'] in (0, .5, .7, .8, .9, .95, .99, selection['selected_threshold']):
            axes[0].annotate(f"t={r['threshold']:g}", (r['sbrad_recall_full_cohort'], r['sbrad_precision']), xytext=(3, 4), textcoords='offset points', fontsize=8)
    axes[0].set(xlabel='SBRAD recall, all 63 true cases', ylabel='SBRAD precision', xlim=(0, 1.02), ylim=(0, 1.02))
    axes[1].plot([r['threshold'] for r in table], [r['sbrad_fp'] for r in table], 'o-', label='False SBRAD alerts')
    axes[1].plot([r['threshold'] for r in table], [r['abstained'] for r in table], 's-', label='ABSTAIN records')
    axes[1].set(xlabel='SBRAD threshold', ylabel='Number of validation ECGs'); axes[1].legend()
    for ax in axes: ax.grid(alpha=.25)
    fig.suptitle('Frozen model — fold 9 only; validation-selected rule, not medical validation')
    save(fig, 'SBRAD_PRECISION_RECALL_PLOT.png')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    bins = np.linspace(0, 1, 21)
    for group, mask in masks.items():
        axes[0].hist(prob[mask, 3], bins=bins, histtype='step', linewidth=2, label=f'{group} (n={int(mask.sum())})')
        scores = np.sort(prob[mask, 3])
        if len(scores): axes[1].step(scores, np.arange(1, len(scores)+1)/len(scores), where='post', label=group)
    axes[0].set(xlabel='Softmax p(SBRAD), not assumed calibrated', ylabel='ECG count')
    axes[1].set(xlabel='Softmax p(SBRAD)', ylabel='Within-group cumulative fraction', ylim=(0, 1))
    for ax in axes: ax.legend(fontsize=8); ax.grid(alpha=.25); ax.axvline(selection['selected_threshold'], color='black', linestyle='--', alpha=.5)
    save(fig, 'SBRAD_CONFIDENCE_DISTRIBUTIONS.png')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout='constrained')
    for ax, r, title in zip(axes, (rows[0], selected), ('Original argmax', f"SBRAD threshold {selection['selected_threshold']:g} + ABSTAIN")):
        cm = np.asarray(r['confusion_matrix_with_abstain'])
        ax.imshow(cm, cmap='Blues', vmin=0, vmax=cm.max())
        for i in range(4):
            for j in range(5): ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='white' if cm[i,j] > cm.max()/2 else 'black')
        ax.set(xticks=range(5), xticklabels=(*CLASS_NAMES, 'UNCERTAIN'), yticks=range(4), yticklabels=CLASS_NAMES,
               xlabel='Decision', ylabel='True metadata label', title=title)
        ax.tick_params(axis='x', labelrotation=30)
    save(fig, 'VALIDATION_CONFUSION_MATRICES.png')
    after_stats = {str(p): [p.stat().st_size, p.stat().st_mtime_ns]
                   for root in ('data/processed/ecg_dataset_v1', 'data/processed/ptbxl_rhythm_v1')
                   for p in Path(root).rglob('*') if p.is_file()}
    assert dataset_stats == after_stats, 'Exported datasets changed'
    assert all(file_sha256(p) == sha for p, sha in protected.items()), 'Protected file changed'
    assert file_sha256(checkpoint) == checkpoint_sha
    assert len(index.read_ecg_ids) == 1958 and set(index.read_ecg_ids) == set(data['ecg_id'])
    summary = {'status': 'passed', 'validation_records': len(y), 'validation_patients': len(set(data['patient_id'].tolist())),
        'checkpoint_sha256_before_and_after': checkpoint_sha, 'model_state_unchanged': True,
        'phase18_predictions_reproduced': True, 'waveform_folds_read': [9], 'test_waveform_reads': 0,
        'blocked_waveform_reads': index.blocked_reads, 'inference_wall_seconds': elapsed,
        'protected_files_unchanged': len(protected), 'dataset_files_size_mtime_unchanged': len(dataset_stats),
        'true_vs_false_sbrad_alert_ranking_auc': ranking_auc,
        'baseline_sbrad': rows[0]['per_class'][3], 'selection': selection, 'selected_metrics': selected,
        'torch': str(torch.__version__), 'numpy': np.__version__,
        'confusion_axes': 'true rows: SR,AFIB,STACH,SBRAD; columns: SR,AFIB,STACH,SBRAD,ABSTAIN'}
    write_json(output / 'analysis_summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('results/model_evaluation/phase18b_validation'))
    parser.add_argument('--checkpoint', type=Path, default=Path('results/model_training/phase18_full/run_001/best_model.pt'))
    args = parser.parse_args()
    analyze(args.output_dir, args.checkpoint)
