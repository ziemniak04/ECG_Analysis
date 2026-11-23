"""Reference metric values include ties, missing classes and honest undefined cases."""
import numpy as np
import pytest
from src.ptbxl_metrics import binary_average_precision,classification_metrics


def test_perfect_four_class_predictions():
    y=np.arange(4);p=np.full((4,4),.01);p[np.arange(4),y]=.97
    m=classification_metrics(y,p)
    assert m['accuracy']==m['macro_f1']==m['macro_auprc']==1
    assert m['confusion_matrix']==np.eye(4,dtype=int).tolist()
    assert all(row['recall']==row['specificity']==row['precision']==row['f1']==1 for row in m['per_class'])


def test_all_majority_confusion_and_metrics_have_known_values():
    m=classification_metrics(np.arange(4),np.tile([.7,.1,.1,.1],(4,1)))
    assert m['accuracy']==.25 and m['macro_f1']==pytest.approx(.1) and m['macro_auprc']==.25
    assert m['confusion_matrix']==[[1,0,0,0]]*4
    assert m['per_class'][0]['precision']==.25 and m['per_class'][0]['recall']==1
    assert m['per_class'][0]['specificity']==0 and m['per_class'][0]['f1']==.4
    assert m['per_class'][1]['precision'] is None
    assert m['per_class'][1]['recall']==0 and m['per_class'][1]['f1']==0


def test_average_precision_step_integral_not_trapezoid():
    assert binary_average_precision([1,0,1,0],[.9,.8,.7,.1])==pytest.approx(5/6)


def test_probability_ties_grouped_invariant_to_input_order():
    assert binary_average_precision([1,0],[.5,.5])==.5
    assert binary_average_precision([0,1],[.5,.5])==.5


def test_absent_classes_and_single_class_specificity_are_null():
    m=classification_metrics(np.array([0,0]),np.tile([.7,.1,.1,.1],(2,1)))
    assert m['macro_f1'] is None and m['macro_f1_over_available_classes']==1
    assert m['macro_auprc'] is None and m['macro_auprc_over_available_classes'] is None
    assert m['absent_true_classes']==['AFIB','STACH','SBRAD']
    assert m['per_class'][0]['specificity'] is None
    assert m['per_class'][1]['recall'] is None and m['per_class'][1]['f1'] is None
    assert binary_average_precision([0,0],[.1,.2]) is None


@pytest.mark.parametrize('y,p',[
    ([],np.empty((0,4))),([4],[[.25]*4]),([.5],[[.25]*4]),([0],[[float('nan'),0,0,1]]),
    ([0],[[.1,.1,.1,.1]]),([0],[[1,0,0]]),([0],[[1.1,-.1,0,0]])])
def test_invalid_metric_inputs_rejected(y,p):
    with pytest.raises(ValueError):classification_metrics(np.asarray(y),p)
