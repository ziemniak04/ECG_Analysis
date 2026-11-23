"""Read-only post-training verification; never evaluates a waveform or fold 10."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src import ptbxl_training as training
from src.ptbxl_full_run import APPROVED
from src.ptbxl_metrics import classification_metrics
from src.ptbxl_protocol import file_sha256


def validate(output, audit):
    output, audit = Path(output), Path(audit)
    result = json.loads((output / 'result.json').read_text())
    assert result['status'] in ('completed', 'early_stopped'), result
    assert not result['test_fold_used']
    assert result['checkpoint_reload_identical_logits']
    assert result['parameters_changed_this_run']
    assert json.loads((output / 'training_config.json').read_text()) == APPROVED
    checkpoint = training.load_checkpoint(output / 'last_resumable_checkpoint.pt')
    history, plan = checkpoint['history'], checkpoint['plan']
    assert len(history) == result['completed_epochs'] == checkpoint['completed_epoch']
    assert checkpoint['optimizer_steps'] == len(history) * 981
    for row in history:
        for partition, expected, batches, folds in (
            ('train', 15689, 981, set(range(1, 9))),
            ('validation', 1958, 123, {9}),
        ):
            metrics = row[partition]
            assert metrics['record_count'] == expected
            assert metrics['batches'] == batches
            assert set(metrics['strat_folds']) <= folds
            assert len(set(metrics['ecg_ids'])) == expected
            assert set(metrics['ecg_ids']) == set(plan['partitions'][partition]['ecg_ids'])
            assert np.isfinite(metrics['loss'])
            recomputed = classification_metrics(
                np.asarray(metrics['true_labels'], dtype=np.int64), metrics['probabilities'])
            for key in ('confusion_matrix', 'per_class', 'accuracy', 'macro_f1', 'macro_auprc'):
                assert recomputed[key] == metrics[key], (row['epoch'], partition, key)
    best_row = max(history, key=lambda row: row['validation']['macro_f1'])
    assert best_row['epoch'] == result['best_epoch'] == checkpoint['best_epoch']
    assert checkpoint['early_stopped'] == (result['status'] == 'early_stopped')
    if checkpoint['early_stopped']:
        assert checkpoint['bad_epochs'] >= APPROVED['patience']
    best = torch.load(output / 'best_model.pt', map_location='cpu', weights_only=True)
    for name, tensor in checkpoint['best_model_state_dict'].items():
        assert torch.equal(tensor, best['model_state_dict'][name])
    protected = json.loads((audit / 'preservation_before.json').read_text())
    changed = [name for name, digest in protected.items()
               if not Path(name).is_file() or file_sha256(name) != digest]
    assert not changed, changed
    before = json.loads((audit / 'dataset_stats_before.json').read_text())
    after = {str(p): [p.stat().st_size, p.stat().st_mtime_ns]
             for root in ('data/processed/ecg_dataset_v1', 'data/processed/ptbxl_rhythm_v1')
             for p in Path(root).rglob('*') if p.is_file()}
    assert before == after, 'Exported dataset files changed'
    execution = json.loads((audit / 'execution_summary.json').read_text())
    assert execution['blocked_test_reads'] == 0
    assert '10' not in execution['data_read_counts_by_fold']
    assert json.loads((audit / 'preflight.json').read_text())['verified_npz'] == 17647
    hashes = {str(p.relative_to(output)): file_sha256(p)
              for p in output.rglob('*') if p.is_file()}
    report = {
        'status': 'passed', 'epochs_verified': len(history),
        'training_records_per_epoch': 15689, 'validation_records_per_epoch': 1958,
        'test_waveforms_read': 0, 'dataset_files_unchanged': len(before),
        'protected_files_unchanged': len(protected),
        'best_epoch': best_row['epoch'], 'artifact_sha256': hashes,
        'metrics_recomputed_from_saved_predictions': True,
        'checkpoint_best_state_matches': True,
        'best_validation_metrics': {k: best_row['validation'][k] for k in
            ('accuracy', 'macro_f1', 'macro_auprc', 'per_class', 'confusion_matrix')},
    }
    training.write_json_new(audit / 'final_integrity.json', report)
    return {k: v for k, v in report.items() if k != 'artifact_sha256'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('results/model_training/phase18_full/run_001'))
    parser.add_argument('--audit-dir', type=Path, default=Path('results/model_training/phase18_full/audit_run_001'))
    args = parser.parse_args()
    print(json.dumps(validate(args.output_dir, args.audit_dir), indent=2))
