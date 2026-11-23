"""Controlled ablation mechanics; synthetic fixtures never establish performance."""
from dataclasses import asdict
import json
from types import SimpleNamespace
import numpy as np
import pytest
from src import ptbxl_training as trainer
from src.ptbxl_optimization import (configuration, partial_training_weights,
    controlled_loss, select_candidate, ResourceGuard)
from src.ptbxl_protocol import training_class_weights
from test_ptbxl_training import fixture_index


def test_only_loss_differs_between_full_configurations():
    a=asdict(configuration('B_unweighted')); b=asdict(configuration('C_sqrt_balanced'))
    assert {k for k in a if a[k]!=b[k]}=={'class_weighted_loss'}
    assert a['seed']==42 and a['batch_size']==16 and a['epochs']==15
    assert a['train_per_class'] is None and a['validation_per_class'] is None
    with pytest.raises(ValueError): configuration('A_balanced')


def test_partial_weights_use_only_training_and_square_root(fixture_index):
    indices=(0,1,2,3,4,5,6)  # All four classes, deliberately unequal counts.
    balanced=training_class_weights(fixture_index,indices).astype(np.float64)
    expected=np.sqrt(balanced);expected/=expected.mean()
    actual=partial_training_weights(fixture_index,indices)
    np.testing.assert_allclose(actual,expected,rtol=1e-7)
    assert actual.dtype==np.float32 and actual.mean()==pytest.approx(1.)
    with pytest.raises(ValueError): partial_training_weights(fixture_index,(0,1,2,3,8))
    with pytest.raises(ValueError): partial_training_weights(fixture_index,(0,1,2,3,12))


def test_loss_adapter_restores_original_functions_and_fingerprints(fixture_index):
    original_plan,original_weights=trainer.build_plan,trainer.training_class_weights
    config=configuration('C_sqrt_balanced')
    with controlled_loss('C_sqrt_balanced'):
        a=trainer.build_plan(fixture_index,config)
        b=trainer.build_plan(fixture_index,config)
        assert a==b
        np.testing.assert_array_equal(a['class_weights'],partial_training_weights(fixture_index,fixture_index.partitions.train))
        assert a['loss_ablation']['variant']=='C_sqrt_balanced'
        assert 'ptbxl_optimization.py' in a['code_sha256']
    assert trainer.build_plan is original_plan and trainer.training_class_weights is original_weights
    with pytest.raises(RuntimeError):
        with controlled_loss('B_unweighted'):
            plan=trainer.build_plan(fixture_index,configuration('B_unweighted'))
            assert plan['class_weights'] is None
            raise RuntimeError('test restoration')
    assert trainer.build_plan is original_plan


def candidate(model,threshold,f1=.8,ap=.88,fp=30,afib=.93,sbrad=.8,coverage=1):
    return {'candidate':model+str(threshold),'model':model,'threshold':threshold,
        'AFIB_recall':afib,'STACH_recall':.9,'SBRAD_recall':sbrad,'coverage':coverage,
        'macro_f1_full_cohort':f1,'macro_auprc_full_cohort':ap,'SBRAD_fp':fp,
        'accepted_macro_f1_coverage_conditional':.999}


def test_selection_preserves_afib_and_does_not_rank_conditional_metrics():
    rows=[candidate('A_balanced',0),candidate('B_unweighted',0,f1=.9,afib=.7),
          candidate('C_sqrt_balanced',.65,f1=.83,coverage=.98)]
    result=select_candidate(rows)
    assert result['selected']['model']=='C_sqrt_balanced'
    assert rows[1]['candidate'] not in result['stages']['eligible']


def test_near_ties_prefer_simpler_rule_and_loss():
    rows=[candidate('A_balanced',0),candidate('B_unweighted',0,f1=.798),
          candidate('C_sqrt_balanced',.65,f1=.802,fp=28,coverage=.99)]
    assert select_candidate(rows)['selected']['model']=='B_unweighted'
    rows[1]['SBRAD_recall']=.2
    rows[0]['AFIB_recall']=.1
    rows[2]['coverage']=.9
    assert select_candidate(rows)['status']=='no_eligible_candidate'


def test_resource_guard_stops_before_start_on_excess_rss(tmp_path,monkeypatch):
    import src.ptbxl_optimization as module
    monkeypatch.setattr(module,'system_resources',lambda:{'memory':{'MemAvailable':1000},'disk_free_bytes':1000})
    monkeypatch.setattr(module,'memory_snapshot',lambda:{'rss_bytes':500})
    limits={'maximum_process_rss_bytes':100,'minimum_available_ram_bytes':10,
            'minimum_disk_bytes':10,'maximum_epoch_seconds':100}
    with pytest.raises(RuntimeError,match='RSS'):
        with ResourceGuard(tmp_path,limits):pass


def test_resume_retains_loss_scheme_and_rejects_cross_variant(fixture_index,tmp_path):
    import torch
    torch.set_num_threads(1)
    config=trainer.TrainingConfig(epochs=2,batch_size=4,train_per_class=2,validation_per_class=1,
        max_train_batches=1,max_validation_batches=1,class_weighted_loss=True,seed=42)
    with controlled_loss('C_sqrt_balanced'):
        first=trainer.train_model(config,tmp_path/'first',index=fixture_index,epoch_budget=1)
        resumed=trainer.train_model(config,tmp_path/'resumed',index=fixture_index,
            resume_from=first['last_completed_checkpoint'])
        full=trainer.train_model(config,tmp_path/'full',index=fixture_index)
    assert resumed['completed_epochs']==2
    a=trainer.load_checkpoint(resumed['last_completed_checkpoint'])
    b=trainer.load_checkpoint(full['last_completed_checkpoint'])
    for key,value in a['model_state_dict'].items(): assert torch.equal(value,b['model_state_dict'][key])
    unweighted=trainer.TrainingConfig(epochs=2,batch_size=4,train_per_class=2,validation_per_class=1,
        max_train_batches=1,max_validation_batches=1,class_weighted_loss=False,seed=42)
    with controlled_loss('B_unweighted'),pytest.raises(ValueError,match='Resume plan'):
        trainer.train_model(unweighted,tmp_path/'bad_resume',index=fixture_index,
            resume_from=first['last_completed_checkpoint'])
