"""No real held-out I/O or inference: exclusive-claim and synthetic contracts."""
import json
import numpy as np
import pytest
import torch
from src.mitbih_final_test import claim_once,SinglePassEngine,ecg_contract
from src.mitbih_rr import rr_features,transform_rr


def test_claim_is_exclusive_and_keeps_original(tmp_path):
    p=tmp_path/'claim.json';claim_once(p,{'checkpoint':'frozen'})
    with pytest.raises(FileExistsError):claim_once(p,{'checkpoint':'other'})
    assert json.loads(p.read_text())=={'checkpoint':'frozen'}
    assert p.stat().st_mode&0o222==0


class FakeModel(torch.nn.Module):
    def __init__(self):super().__init__();self.layer=torch.nn.Linear(6,5);self.calls=0
    def forward(self,x,rr):
        assert not self.training and not torch.is_grad_enabled()
        self.calls+=1;return self.layer(rr)


def test_one_pass_per_record_exact_counts_no_duplicate_updates():
    m=FakeModel();engine=SinglePassEngine(m,{'a':3,'b':2},2)
    p=engine.record('a',np.zeros((3,2,252),np.float32),np.zeros((3,6),np.float32))
    assert p.shape==(3,5) and np.isfinite(p).all()
    assert m.calls==2
    with pytest.raises(PermissionError):engine.record('a',None,None)
    assert m.calls==2
    with pytest.raises(ValueError):engine.finish()
    engine.record('b',np.zeros((2,2,252),np.float32),np.zeros((2,6),np.float32))
    assert engine.finish()['beats']==5 and engine.finish()['parameters_unchanged']
    assert m.calls==3


def test_engine_rejects_invalid_shapes_before_forward():
    m=FakeModel();engine=SinglePassEngine(m,{'a':1})
    with pytest.raises(ValueError):engine.record('a',np.zeros((1,2,251),np.float32),np.zeros((1,6),np.float32))
    assert m.calls==0


def test_record_specific_normalization_is_not_global_refit():
    a={'normalization':dict(offset=[0,1],scale=[2,3],source_ids=['100'],channel_names=['MLII','V1'],sample_count=100,method='zscore',scope='recording'),'sequence_length':252}
    b=json.loads(json.dumps(a));b['normalization']['source_ids']=['112'];b['normalization']['offset']=[2,4]
    assert ecg_contract(a)==ecg_contract(b)
    b['normalization']['method']='minmax'
    assert ecg_contract(a)!=ecg_contract(b)


def test_frozen_past_rr_has_no_future_or_cross_record_context():
    a=np.array([10,370,730,1000],dtype=np.int64)
    raw,valid=rr_features(a,a,360)
    short,sv=rr_features(a[:3],a[:3],360)
    np.testing.assert_array_equal(raw[:3],short);np.testing.assert_array_equal(valid[:3],sv)
    assert not valid[0].any() and valid[1].tolist()==[True,False,False]
    state=dict(schema='mitbih-rr-transform-v1',fitted_partition='train',feature_names=['previous_rr_seconds','preceding_baseline_rr_seconds','previous_over_baseline'],imputation_median=[.8,.8,1.],mean=[.9,.9,1.],scale=[.2,.2,.2])
    frozen=json.dumps(state,sort_keys=True);result=transform_rr(raw,valid,state)
    assert result.shape==(4,6) and result.dtype==np.float32 and np.isfinite(result).all()
    assert json.dumps(state,sort_keys=True)==frozen

from types import SimpleNamespace
from src.mitbih_final_test import AuthorizedTestIO


def test_authorized_scope_blocks_source_writes_and_relocks_after_exit(tmp_path):
    source=tmp_path/'sample.npz';source.write_bytes(b'synthetic')
    claim=tmp_path/'claim.json';claim_once(claim,{'single':'attempt'})
    index=SimpleNamespace(root=tmp_path,by_record={'fake':SimpleNamespace(file='sample.npz')})
    guard=SimpleNamespace(blocked={str(source.resolve())})
    gate=AuthorizedTestIO(guard,index,['fake'],claim)
    with gate:
        assert source.read_bytes()==b'synthetic'
        with pytest.raises(PermissionError):source.open('wb')
        assert source.stat().st_size==9
    with pytest.raises(PermissionError):source.read_bytes()
    assert len(gate.opens)==1 and len(gate.denied)==2


def test_authorized_scope_requires_prior_claim(tmp_path):
    gate=AuthorizedTestIO(SimpleNamespace(blocked=set()),SimpleNamespace(root=tmp_path,by_record={}),[],tmp_path/'missing')
    with pytest.raises(PermissionError):gate.__enter__()
