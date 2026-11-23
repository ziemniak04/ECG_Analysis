import numpy as np
import pytest
from src.mitbih_f_decisions import f_decisions,threshold_metrics,score_summary

P=np.array([[.1,.1,.1,.6,.1],[.1,.1,.1,.5,.2],[.6,.1,.1,.1,.1],[.1,.6,.1,.1,.1],[.1,.1,.6,.1,.1],[.1,.1,.1,.1,.6]])
Y=np.array([3,0,0,1,2,4])

def test_argmax_threshold_equality_and_no_reassignment():
    np.testing.assert_array_equal(f_decisions(P,0),P.argmax(1))
    np.testing.assert_array_equal(f_decisions(P,.6),[3,-1,0,1,2,4])
    np.testing.assert_array_equal(f_decisions(P,.7),[-1,-1,0,1,2,4])
    np.testing.assert_array_equal(P[0],[.1,.1,.1,.6,.1])

def test_counts_recall_denominator_includes_rejections():
    a=threshold_metrics(Y,P,.6)
    assert (a['F_tp'],a['F_fp'],a['F_fn'])==(1,0,0)
    assert a['coverage']==5/6 and a['F_precision']==a['F_recall']==1
    assert a['N_abstentions']==1 and a['N_abstention_rate']==.5
    b=threshold_metrics(Y,P,.7)
    assert b['F_precision'] is None and b['F_recall']==0 and b['F_fn']==1
    assert b['accepted_only_macro_f1'] is None and b['accepted_absent_classes']==['F']
    assert b['accepted_only_accuracy']==1 and b['F_abstention_rate']==1

@pytest.mark.parametrize('t',[-1,1.1,True,float('nan')])
def test_invalid_threshold(t):
    with pytest.raises(ValueError):f_decisions(P,t)

def test_invalid_scores_labels_and_determinism():
    with pytest.raises(ValueError):f_decisions(P*2,.5)
    with pytest.raises(ValueError):threshold_metrics(Y.astype(float),P,.5)
    assert threshold_metrics(Y,P,.6)==threshold_metrics(Y,P,.6)
    assert score_summary([])['median'] is None
    assert score_summary([.1,.3,.5])['median']==.3
