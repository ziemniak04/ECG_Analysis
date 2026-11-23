"""Final-test evaluator mechanics on synthetic exports only; never real fold 10."""
from types import SimpleNamespace
import json
import numpy as np
import pytest
import torch
from src.models.ptbxl_cnn import PTBXLRhythmCNN
from src.ptbxl_final_evaluation import (TestOnlyIndex,claim_once,infer_test_batches,
    CHECKPOINT_SHA256,CHECKPOINT)
from src.ptbxl_protocol import PTBXLExportIndex,file_sha256
from src.ptbxl_metrics import classification_metrics
from test_ptbxl_training import fixture_index


def test_approved_checkpoint_identity_without_waveform_access():
    assert file_sha256(CHECKPOINT)==CHECKPOINT_SHA256


def test_claim_is_exclusive_and_never_overwrites(tmp_path):
    claim=tmp_path/'claim.json'
    claim_once(claim,{'checkpoint':'frozen','rule':'argmax'})
    before=claim.read_bytes()
    with pytest.raises(FileExistsError):claim_once(claim,{'checkpoint':'other'})
    assert claim.read_bytes()==before


@pytest.mark.parametrize('fold',[1,8,9])
def test_non_test_folds_rejected_before_io(monkeypatch,fold):
    index=object.__new__(TestOnlyIndex)
    index.entries=(SimpleNamespace(strat_fold=fold),);index.blocked_reads=0
    def forbidden(*args,**kwargs):raise AssertionError('Reader must not run')
    monkeypatch.setattr(PTBXLExportIndex,'load_numpy',forbidden)
    with pytest.raises(RuntimeError,match='test fold 10'):index.load_numpy(0)
    assert index.blocked_reads==1


@pytest.mark.parametrize('position',[-1,True,1.5,5])
def test_invalid_positions_rejected(position):
    index=object.__new__(TestOnlyIndex);index.entries=(SimpleNamespace(strat_fold=10),)
    with pytest.raises(IndexError):index.load_numpy(position)


def guarded_fixture(fixture_index):
    return TestOnlyIndex(fixture_index.root,fixture_index.manifest_path,
        require_full_cohort=False,verify_source_metadata=False)


def test_single_pass_shape_argmax_no_abstention_and_unchanged_model(fixture_index):
    index=guarded_fixture(fixture_index);torch.set_num_threads(1);torch.manual_seed(42)
    model=PTBXLRhythmCNN()
    before={k:v.clone() for k,v in model.state_dict().items()}
    calls=[]
    def inspect(module,args,result):
        assert not module.training and torch.is_inference_mode_enabled()
        calls.append(args[0].shape[0])
        assert tuple(result.shape)==(len(args[0]),4)
    hook=model.register_forward_hook(inspect)
    saved=[]
    arrays,timing=infer_test_batches(model,index,batch_size=3,
        batch_sink=lambda n,c,t:saved.append((n,c['ecg_id'].tolist())))
    hook.remove()
    assert calls==[3,1] and len(saved)==2  # No warmup or reload forward.
    assert arrays['ecg_id'].tolist()==[90013,90014,90015,90016]
    assert index.read_ecg_ids==arrays['ecg_id'].tolist()
    assert set(arrays['strat_fold'].tolist())=={10}
    assert arrays['probabilities'].shape==(4,4) and np.isfinite(arrays['probabilities']).all()
    np.testing.assert_array_equal(arrays['predicted_labels'],arrays['probabilities'].argmax(axis=1))
    assert np.isin(arrays['predicted_labels'],[0,1,2,3]).all()
    assert timing['abstentions']==0 and timing['coverage']==1
    assert timing['waveform_passes']==1 and timing['warmup_test_forwards']==0
    assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())
    assert all(p.grad is None for p in model.parameters())


def test_invalid_logits_fail_without_silent_record_skip(fixture_index):
    class Invalid(torch.nn.Module):
        def forward(self,x):return torch.full((len(x),4),float('nan'))
    with pytest.raises(ValueError,match='logits'):
        infer_test_batches(Invalid(),guarded_fixture(fixture_index))


def test_no_threshold_or_second_choice_is_used(fixture_index):
    class Fixed(torch.nn.Module):
        def forward(self,x):
            # SBRAD wins with probability below .65 and MUST remain SBRAD.
            return torch.tensor([[0.,0.,0.,.1]]).repeat(len(x),1)
    arrays,_=infer_test_batches(Fixed(),guarded_fixture(fixture_index))
    assert (arrays['probabilities'][:,3]<.65).all()
    assert arrays['predicted_labels'].tolist()==[3,3,3,3]
    metrics=classification_metrics(arrays['labels'],arrays['probabilities'])
    assert metrics['per_class'][3]['tp']==1 and metrics['per_class'][3]['fp']==3
    assert metrics['per_class'][3]['recall']==1 and metrics['per_class'][3]['precision']==.25


def test_real_claim_blocks_reexecution_before_index_or_model(monkeypatch,tmp_path):
    import src.ptbxl_final_evaluation as module
    claim=tmp_path/'claim.json';claim_once(claim,{'claimed':True})
    monkeypatch.setattr(module,'ONCE_CLAIM',claim)
    def forbidden(*args,**kwargs):raise AssertionError('Must fail before dataset/model initialization')
    monkeypatch.setattr(module,'TestOnlyIndex',forbidden)
    with pytest.raises(FileExistsError,match='already claimed'):module.evaluate_once(tmp_path/'unused')
