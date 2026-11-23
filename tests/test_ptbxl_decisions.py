"""Decision-rule mechanics on controlled fixtures; no final-test ECG access."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from src.models.ptbxl_cnn import PTBXLRhythmCNN
from src.ptbxl_decisions import (sbrad_decisions, decision_metrics, threshold_study,
    recommend_threshold, ValidationOnlyIndex, frozen_validation_predictions)
from src.ptbxl_protocol import PTBXLExportIndex
from src.ptbxl_metrics import classification_metrics


@pytest.fixture
def scores():
    return np.array([[.05,.05,.05,.85], [.1,.1,.1,.7], [.7,.05,.05,.2],
                     [.1,.7,.1,.1], [.1,.1,.7,.1], [.25,.25,.25,.25]])


def test_threshold_rejects_only_sbrad_winner_and_never_reassigns(scores):
    original = scores.copy()
    assert sbrad_decisions(scores, .8).tolist() == [3,-1,0,1,2,0]
    np.testing.assert_array_equal(scores, original)
    # SBRAD above threshold but not highest must not become an alert.
    assert sbrad_decisions([[.6,0,0,.4]], .3).tolist() == [0]


def test_equality_is_accepted_and_argmax_ties_match_original(scores):
    assert sbrad_decisions(scores, .7)[1] == 3
    np.testing.assert_array_equal(sbrad_decisions(scores, 0), scores.argmax(axis=1))
    assert sbrad_decisions([[.4,0,.2,.4]], .3).tolist() == [0]


@pytest.mark.parametrize('threshold', [-.1,1.1,float('nan'),float('inf'),True,'0.5'])
def test_invalid_thresholds_rejected(scores, threshold):
    with pytest.raises(ValueError): sbrad_decisions(scores, threshold)


@pytest.mark.parametrize('prob', [[], [[1,0,0]], [[.1,.1,.1,.1]], [[-1,1,1,0]], [[0,0,0,float('nan')]]])
def test_invalid_probabilities_rejected(prob):
    with pytest.raises(ValueError): sbrad_decisions(prob, .5)


def test_full_recall_counts_abstained_positives_as_misses():
    y = np.array([3,3,3,0,0,1,2])
    pred = np.array([3,-1,0,3,-1,1,2])
    result = decision_metrics(y,pred); s=result['per_class'][3]
    assert (s['tp'],s['fp'],s['fn'],s['tn']) == (1,1,2,3)
    assert s['precision'] == .5 and s['recall'] == pytest.approx(1/3)
    assert s['specificity'] == .75 and s['f1'] == .4
    assert result['abstained'] == 2 and result['coverage'] == pytest.approx(5/7)
    assert result['accepted_accuracy_coverage_conditional'] == .6
    assert result['per_class'][0]['abstention_rate'] == .5
    assert result['per_class'][3]['abstention_rate'] == pytest.approx(1/3)
    assert sum(map(sum,result['confusion_matrix_with_abstain'])) == len(y)


def test_argmax_metrics_match_existing_implementation(scores):
    y=np.array([3,0,0,1,2,3])
    result=decision_metrics(y,sbrad_decisions(scores,0))
    old=classification_metrics(y,scores)
    assert result['accepted_accuracy_coverage_conditional'] == old['accuracy']
    assert result['accepted_macro_f1_coverage_conditional'] == old['macro_f1']
    for new,previous in zip(result['per_class'],old['per_class']):
        for key in ('tp','fp','fn','tn','precision','recall','specificity','f1'):
            assert new[key] == previous[key]


def test_all_abstain_and_missing_class_metrics_are_explicit():
    result=decision_metrics(np.array([0,3]),np.array([-1,-1]))
    assert result['coverage']==0 and result['accepted_accuracy_coverage_conditional'] is None
    assert result['accepted_macro_f1_coverage_conditional'] is None
    assert result['per_class'][3]['recall']==0 and result['per_class'][3]['precision'] is None
    assert result['per_class'][1]['abstention_rate'] is None


@pytest.mark.parametrize('y,pred', [([],[]), ([4],[0]), ([0.],[0]), ([0],[4]), ([0],[0,1])])
def test_invalid_metric_inputs(y,pred):
    with pytest.raises(ValueError): decision_metrics(y,pred)


def test_threshold_study_is_deterministic_and_monotonic(scores):
    y=np.array([3,0,0,1,2,3])
    a=threshold_study(y,scores); assert a==threshold_study(y,scores)
    assert [r['abstained'] for r in a] == sorted(r['abstained'] for r in a)
    assert [r['per_class'][3]['fp'] for r in a] == sorted((r['per_class'][3]['fp'] for r in a),reverse=True)
    assert [r['per_class'][3]['fn'] for r in a] == sorted(r['per_class'][3]['fn'] for r in a)


def test_selection_does_not_sacrifice_recall_for_precision():
    # A high-precision candidate that loses nearly all true cases is ineligible.
    def row(t,tp,fp,coverage):
        return {'threshold':t,'coverage':coverage,'per_class':[{}, {}, {},
            {'tp':tp,'fp':fp,'f1':2*tp/(100+tp+fp)}]}
    rows=[row(0,100,100,1), row(.5,95,60,.98), row(.9,20,0,.98)]
    selected=recommend_threshold(rows)
    assert selected['selected_threshold']==.5
    assert .9 not in selected['eligible_thresholds']
    rows[1]['coverage']=.95
    assert recommend_threshold(rows)['selected_threshold']==0


@pytest.mark.parametrize('fold',[1,8,10])
def test_nonvalidation_reads_blocked_before_io(monkeypatch,fold):
    index=object.__new__(ValidationOnlyIndex)
    index.entries=(SimpleNamespace(strat_fold=fold),); index.blocked_reads=0;index.read_ecg_ids=[]
    def forbidden(*args,**kwargs): raise AssertionError('Reader must not be reached')
    monkeypatch.setattr(PTBXLExportIndex,'load_numpy',forbidden)
    with pytest.raises(RuntimeError,match='fold 9'): index.load_numpy(0)
    assert index.blocked_reads==1 and index.read_ecg_ids==[]


def test_frozen_inference_uses_only_validation_and_preserves_parameters(monkeypatch):
    index=object.__new__(ValidationOnlyIndex)
    index.entries=(SimpleNamespace(strat_fold=10,ecg_id=900),
                   SimpleNamespace(strat_fold=9,ecg_id=901,patient_id=10,label=3))
    index.partitions=SimpleNamespace(validation=(1,),test=(0,))
    index.read_ecg_ids=[];index.blocked_reads=0
    def fixture_reader(self,pos,**kwargs):
        assert pos==1
        return {'X':np.ones((12,1000),dtype=np.float32),'y':np.array(3,dtype=np.int64)}
    monkeypatch.setattr(PTBXLExportIndex,'load_numpy',fixture_reader)
    torch.set_num_threads(1)
    model=PTBXLRhythmCNN()
    before={k:v.clone() for k,v in model.state_dict().items()}
    result=frozen_validation_predictions(model,index)
    assert result['ecg_id'].tolist()==[901] and result['strat_fold'].tolist()==[9]
    assert index.read_ecg_ids==[901] and index.blocked_reads==0
    assert not model.training and all(p.grad is None for p in model.parameters())
    for name,value in model.state_dict().items(): assert torch.equal(value,before[name])
    second=frozen_validation_predictions(model,index)
    np.testing.assert_array_equal(result['probabilities'],second['probabilities'])
