"""Reporting replay only: no waveform loading or neural-network execution."""
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from src.mitbih_final_reporting import collect_saved_predictions,metrics_from_saved


def fixture_predictions(tmp_path,*,duplicate=False,bad_argmax=False):
    (tmp_path/'predictions').mkdir();y=np.tile(np.arange(5,dtype=np.int64),2);n=len(y)
    pred=y.copy()
    if bad_argmax:pred[0]=1
    ids=np.array([f'fixture:{i}' for i in range(n)])
    if duplicate:ids[-1]=ids[0]
    np.savez(tmp_path/'predictions/a_predictions.npz',probabilities=np.eye(5)[y],y=y,predicted_indices=pred,record_ids=np.array(['a']*n),patient_groups=np.array(['g1']*n),beat_ids=ids,global_source_indices=np.arange(n))
    return dict(summary={'test':dict(record_ids=['a'],beats=n,class_counts=dict.fromkeys('NSVFQ',2))},split={'groups':{'test':['g1']}})


def test_saved_metrics_need_no_model_call(tmp_path,monkeypatch):
    protocol=fixture_predictions(tmp_path)
    def forbidden(*a,**kw):raise AssertionError('No inference in reporting')
    monkeypatch.setattr(torch.nn.Module,'__call__',forbidden)
    a=collect_saved_predictions(tmp_path,protocol);m,g=metrics_from_saved(a)
    assert m['macro_f1']==m['macro_auprc']==m['accuracy']==1 and len(g)==1

@pytest.mark.parametrize('kwargs',[dict(duplicate=True),dict(bad_argmax=True)])
def test_duplicate_or_changed_saved_decisions_rejected(tmp_path,kwargs):
    protocol=fixture_predictions(tmp_path,**kwargs)
    with pytest.raises(ValueError):collect_saved_predictions(tmp_path,protocol)


def test_real_saved_results_reproduce_metrics_without_inference(monkeypatch):
    root=Path('results/model_evaluation/phase26_mitbih_final_test/run_001')
    if not (root/'final_test_metrics.json').exists():pytest.skip('Final evaluation not present; never generate it from a test')
    def forbidden(*a,**kw):raise AssertionError('Final NN inference must not repeat')
    monkeypatch.setattr(torch.nn.Module,'__call__',forbidden)
    protocol=json.loads((root/'pre_evaluation_manifest.json').read_text());expected=json.loads((root/'final_test_metrics.json').read_text())
    a=collect_saved_predictions(root,protocol);m,g=metrics_from_saved(a)
    assert m['beat_count']==23392 and len(g)==9 and m['confusion_matrix']==expected['confusion_matrix']
    for k in ('accuracy','macro_f1','macro_auprc'):assert m[k]==expected[k]
