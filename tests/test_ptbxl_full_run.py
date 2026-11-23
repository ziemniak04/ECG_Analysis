"""Execution audit hooks; no real training, checkpoint-B test-fold access or data writes."""
from types import SimpleNamespace
import json
import logging
import pytest
from src import ptbxl_training as trainer
from src.ptbxl_full_run import GuardedIndex,atomic_status,epoch_timing,APPROVED
from src.ptbxl_training import TrainingConfig


def test_approved_configuration_matches_frozen_json():
    from pathlib import Path
    config=json.loads(Path('configs/ptbxl_phase18_full_v1.json').read_text())
    assert config==APPROVED
    assert TrainingConfig(**config).validation_per_class is None
    assert config['max_validation_batches'] is None and config['class_weighted_loss']


def test_guard_rejects_test_waveform_before_reader():
    # Metadata-only object: deliberately no actual waveform access.
    index=object.__new__(GuardedIndex)
    index.entries=(SimpleNamespace(strat_fold=10),)
    index.blocked_test_reads=0;index.read_counts={}
    with pytest.raises(RuntimeError,match='Final-test'):index.load_numpy(0)
    assert index.blocked_test_reads==1 and index.read_counts=={}


def test_atomic_status_updates_only_current_owned_status(tmp_path):
    atomic_status(tmp_path,status='preflight',epoch=0)
    atomic_status(tmp_path,status='training',epoch=1)
    assert json.loads((tmp_path/'live_status.json').read_text())['epoch']==1
    assert len(list(tmp_path.iterdir()))==1


def test_timing_hooks_delegate_and_restore_without_changing_metrics(tmp_path,monkeypatch):
    def loader(*args,**kwargs):return 'loader'
    def epoch(*args,**kwargs):return {'loss':.5,'unchanged':True}
    monkeypatch.setattr(trainer,'make_loader',loader);monkeypatch.setattr(trainer,'epoch_pass',epoch)
    monkeypatch.setattr(trainer.LOG,'level',logging.INFO)
    index=SimpleNamespace(read_counts={1:4,9:4},blocked_test_reads=0)
    with epoch_timing(tmp_path,tmp_path/'unused',index):
        assert trainer.make_loader(index,(), 'train',None,1)=='loader'
        assert trainer.make_loader(index,(), 'validation',None,1)=='loader'
        assert trainer.epoch_pass(None,partition='train')=={'loss':.5,'unchanged':True}
        trainer.epoch_pass(None,partition='validation')
        trainer.LOG.info('epoch %s: train loss %.6f val loss %.6f val Macro F1 %.6f (%s updates)',1,.5,.4,.2,1)
    assert trainer.make_loader is loader and trainer.epoch_pass is epoch
    row=json.loads((tmp_path/'epoch_times.jsonl').read_text())
    assert row['epoch']==1 and row['epoch_seconds']>=0
    assert row['train_seconds']>=0 and row['validation_seconds']>=0
    assert json.loads((tmp_path/'live_status.json').read_text())['status']=='epoch_completed'
