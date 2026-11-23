"""Two-input training mechanics on synthetic shards, under held-out access guards."""
import json
from pathlib import Path
from dataclasses import replace
import numpy as np
import pytest
import torch
from test_mitbih_rr import rr_fixture,fixture_export
from src.mitbih_execution_guard import HoldoutAccessGuard,require_guard
from src.mitbih_training import MITBIHTrainingConfig
from src.mitbih_cnn_rr_training import train_model,build_plan,load_checkpoint,epoch_pass,make_loader
from src.mitbih_rr_dataset import MITBIHRRDataset
from src.models.mitbih_cnn_rr import MITBIHBeatCNNRR


def test_production_path_requires_active_guard(tmp_path):
    with pytest.raises(PermissionError):require_guard()
    with pytest.raises(PermissionError):train_model(MITBIHTrainingConfig(),tmp_path/'never',allow_full_training=True)
    assert not (tmp_path/'never').exists()


def test_runtime_guard_blocks_all_real_paths_before_any_os_open(tmp_path):
    with HoldoutAccessGuard(tmp_path/'guard.json') as guard:
        result=guard.probe()
        assert result==dict(blocked_paths=45,expected_denials=45,actual_reads=0)
        assert require_guard() is guard and not guard.unexpected
    logged=json.loads((tmp_path/'guard.json').read_text())
    assert logged['unexpected_denial_count']==0 and logged['exit_exception'] is None


def test_full_two_input_population_weighted_loss_and_metadata(rr_fixture,tmp_path):
    index,split,root=rr_fixture;config=MITBIHTrainingConfig(epochs=2,batch_size=8)
    with HoldoutAccessGuard(tmp_path/'guard.json'):
        result=train_model(config,tmp_path/'run',index=index,split=split,rr_root=root,allow_full_training=True)
        checkpoint=torch.load(result['last_checkpoint'],weights_only=True)
        assert result['completed_epochs']==2
        for row in checkpoint['history']:
            assert row['train']['beat_count']==split['summary']['train']['beats']
            assert row['validation']['beat_count']==split['summary']['validation']['beats']
            assert row['validation']['complete_partition']
            assert row['validation']['loss']==row['validation']['loss_numerator']/row['validation']['loss_denominator']
            assert sum(row['validation']['class_prediction_distribution'].values())==row['validation']['beat_count']
        saved=torch.load(tmp_path/'run/best_model.pt',weights_only=True)
        assert saved['model_configuration']['parameter_count']==17989
        assert saved['rr_manifest_sha256']==checkpoint['plan']['rr_manifest_sha256']
        assert saved['preprocessing_configuration']['RR_transform']['fitted_partition']=='train'
        assert set(index.read_records)<=set(split['summary']['train']['record_ids']+split['summary']['validation']['record_ids'])


def test_resume_exact_and_reject_wrong_rr_plan(rr_fixture,tmp_path):
    index,split,root=rr_fixture;config=MITBIHTrainingConfig(epochs=3,batch_size=16)
    with HoldoutAccessGuard(tmp_path/'guard.json'):
        full=train_model(config,tmp_path/'full',index=index,split=split,rr_root=root,allow_full_training=True)
        partial=train_model(config,tmp_path/'partial',index=index,split=split,rr_root=root,allow_full_training=True,epoch_budget=1)
        resumed=train_model(config,tmp_path/'resume',index=index,split=split,rr_root=root,allow_full_training=True,resume_from=partial['last_checkpoint'])
        a=torch.load(full['last_checkpoint'],weights_only=True);b=torch.load(resumed['last_checkpoint'],weights_only=True)
        for k,v in a['model_state_dict'].items():torch.testing.assert_close(v,b['model_state_dict'][k],rtol=0,atol=0)
        assert full['best_validation_macro_f1']==resumed['best_validation_macro_f1']
        plan=build_plan(index,split,config,root);plan['rr_manifest_sha256']='different'
        with pytest.raises(ValueError,match='identity differs'):load_checkpoint(partial['last_checkpoint'],plan)
        with pytest.raises(FileExistsError):train_model(config,tmp_path/'full',index=index,split=split,rr_root=root,allow_full_training=True)


def test_epoch_rejects_test_and_leaking_group(rr_fixture,tmp_path):
    index,split,root=rr_fixture;config=MITBIHTrainingConfig(batch_size=3)
    with HoldoutAccessGuard(tmp_path/'guard.json'):
        ds=MITBIHRRDataset(root,index=index,split=split,partition='validation')
        batch=next(iter(make_loader(ds,config,1)))
        batch['patient_group'][0]=split['groups']['test'][0]
        with pytest.raises(RuntimeError,match='leakage'):epoch_pass(MITBIHBeatCNNRR(),[batch],split,partition='validation',batch_cap=1)
        with pytest.raises(PermissionError):epoch_pass(MITBIHBeatCNNRR(),[],split,partition='test')
