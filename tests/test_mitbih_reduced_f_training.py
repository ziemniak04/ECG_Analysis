"""One-change controlled experiment checks on small synthetic fixtures only."""
import inspect
from dataclasses import replace
import numpy as np
import pytest
import torch
from test_mitbih_rr import rr_fixture,fixture_export
from src import mitbih_cnn_rr_training as reference
from src import mitbih_reduced_f_training as reduced
from src.mitbih_execution_guard import HoldoutAccessGuard


def test_training_body_is_identical_to_frozen_reference():
    assert inspect.getsource(reference.train_model)==inspect.getsource(reduced.train_model)
    assert reduced.epoch_pass is reference.epoch_pass and reduced.make_loader is reference.make_loader
    assert reduced.load_checkpoint is reference.load_checkpoint


def test_guard_and_explicit_approval_before_io(tmp_path):
    with pytest.raises(PermissionError):reduced.train_model(reduced.MITBIHTrainingConfig(),tmp_path/'never',allow_full_training=True)
    with HoldoutAccessGuard(tmp_path/'guard.json'):
        with pytest.raises(PermissionError):reduced.train_model(reduced.MITBIHTrainingConfig(),tmp_path/'never')
    assert not (tmp_path/'never').exists()


def test_only_F_weight_changed_no_renormalization(rr_fixture,tmp_path):
    index,split,root=rr_fixture;config=reduced.MITBIHTrainingConfig()
    with HoldoutAccessGuard(tmp_path/'guard.json'):
        a=reference.build_plan(index,split,config,root);b=reduced.build_plan(index,split,config,root)
        np.testing.assert_array_equal(np.asarray(b['class_weights'])[[0,1,2,4]],np.asarray(a['class_weights'])[[0,1,2,4]])
        assert b['class_weights'][3]==a['class_weights'][3]*.5
        assert b['loss_weight_adjustment']['renormalized'] is False
        for key in a:
            if key not in ('class_weights','code_sha256'):assert a[key]==b[key]
        for key,value in a['code_sha256'].items():assert b['code_sha256'][key]==value
        for strategy in ('balanced','unweighted'):
            with pytest.raises(ValueError):reduced.build_plan(index,split,replace(config,loss_strategy=strategy),root)
        assert not index.read_records


def test_reduced_weights_used_in_loss_and_exact_resume(rr_fixture,tmp_path):
    index,split,root=rr_fixture;config=reduced.MITBIHTrainingConfig(epochs=2,batch_size=16)
    with HoldoutAccessGuard(tmp_path/'guard.json'):
        full=reduced.train_model(config,tmp_path/'full',index=index,split=split,rr_root=root,allow_full_training=True)
        partial=reduced.train_model(config,tmp_path/'partial',index=index,split=split,rr_root=root,allow_full_training=True,epoch_budget=1)
        resumed=reduced.train_model(config,tmp_path/'resume',index=index,split=split,rr_root=root,allow_full_training=True,resume_from=partial['last_checkpoint'])
        a=torch.load(full['last_checkpoint'],weights_only=True);b=torch.load(resumed['last_checkpoint'],weights_only=True)
        for key,value in a['model_state_dict'].items():assert torch.equal(value,b['model_state_dict'][key])
        weights=a['plan']['class_weights'];counts=split['summary']['validation']['class_counts']
        expected=sum(counts[c]*weights[i] for i,c in enumerate('NSVFQ'))
        for row in a['history']:
            assert row['validation']['complete_partition']
            assert row['validation']['loss_denominator']==pytest.approx(expected)
        assert full['best_validation_macro_f1']==resumed['best_validation_macro_f1']
        with pytest.raises(FileExistsError):reduced.train_model(config,tmp_path/'full',index=index,split=split,rr_root=root,allow_full_training=True)
