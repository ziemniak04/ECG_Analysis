"""Pure diagnostic/selection safeguards, never final-test data."""
import numpy as np
import pytest
import torch
from src.mitbih_cnn_rr_evaluation import neutral_rr,verify_alignment,evaluate


def test_neutral_rr_uses_training_imputation_and_masks():
    state=dict(schema='mitbih-rr-transform-v1',fitted_partition='train',feature_names=['previous_rr_seconds','preceding_baseline_rr_seconds','previous_over_baseline'],imputation_median=[1.,2.,3.],mean=[2.,4.,6.],scale=[2.,4.,6.])
    a=neutral_rr(state,2)
    assert a.dtype==torch.float32 and a.shape==(2,6)
    np.testing.assert_array_equal(a,np.array([[-.5,-.5,-.5,0,0,0]]*2,dtype=np.float32))
    assert torch.equal(a,neutral_rr(state,2))
    with pytest.raises(ValueError):neutral_rr(state,0)


def test_alignment_checks_identity_not_just_length():
    row=dict(record_id='201',patient_group='paired',original_symbol='F',annotation_position=100,source_index=22,channel_0='MLII',channel_1='V1')
    assert verify_alignment([row],[dict(row,beat_id='201:1:100')])
    with pytest.raises(ValueError):verify_alignment([row],[dict(row,record_id='202')])
    with pytest.raises(ValueError):verify_alignment([row],[])


def test_evaluation_requires_execution_guard_before_io():
    with pytest.raises(PermissionError):evaluate('does-not-exist',None,None,None)

from test_mitbih_rr import rr_fixture,fixture_export
from src.mitbih_execution_guard import HoldoutAccessGuard
from src.mitbih_rr_dataset import MITBIHRRDataset
from src.models.mitbih_cnn_rr import MITBIHBeatCNNRR
from src.prepare_dataset import sha256_file


def test_frozen_validation_roundtrip_and_neutral_no_parameter_changes(rr_fixture,tmp_path):
    index,split,root=rr_fixture
    ds=MITBIHRRDataset(root,index=index,split=split,partition='validation')
    model=MITBIHBeatCNNRR();checkpoint=tmp_path/'best.pt'
    torch.save(dict(schema='mitbih-cnn-rr-best-model-v1',class_names=list('NSVFQ'),model_configuration=model.configuration(),rr_manifest_sha256=sha256_file(root/'manifest.json'),preprocessing_configuration={'RR_transform':ds.transform},model_state_dict=model.state_dict()),checkpoint)
    digest=sha256_file(checkpoint);checkpoint.with_suffix('.sha256').write_text(digest)
    with HoldoutAccessGuard(tmp_path/'evaluation_guard.json'):
        first=evaluate(checkpoint,ds,split,None)
        repeated=evaluate(checkpoint,ds,split,None)
        neutral=evaluate(checkpoint,ds,split,None,neutral=True)
        ds.partition='test'
        with pytest.raises(PermissionError):evaluate(checkpoint,ds,split,None)
    np.testing.assert_array_equal(first[3],repeated[3]);np.testing.assert_array_equal(first[4],repeated[4])
    assert first[0]['beat_count']==len(ds) and first[0]['parameters_unchanged']
    assert first[0]['confusion_matrix']==repeated[0]['confusion_matrix']
    assert verify_alignment(first[2],neutral[2])
    assert neutral[0]['RR_condition']=='all_missing_train_median_plus_zero_masks'
    assert sha256_file(checkpoint)==digest
