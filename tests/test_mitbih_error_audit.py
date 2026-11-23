"""Read-only audit tests. No optimizer, backward, fitting, or held-out inference."""
from collections import Counter
import json
from types import SimpleNamespace
import numpy as np
import pytest
from src.mitbih_error_audit import (mapping_rows,select_examples,verify_annotations,require_record,draw_examples)
from src.label_mapping import DEFAULT_MAPPING


def test_intended_mapping_and_symbol_distinctions():
    expected={s:c for c,ss in [('N','NLRej'),('S','AaJS'),('V','VE'),('F','F'),('Q','/fQ')] for s in ss}
    assert dict(DEFAULT_MAPPING)==expected
    counts={'train':Counter({'a':6,'f':56,'F':401}), 'validation':Counter({'a':139,'f':666,'F':38,'+':10})}
    rows={r['symbol']:r for r in mapping_rows(counts,counts)}
    assert rows['a']['project_group']=='S' and rows['f']['project_group']=='Q' and rows['F']['project_group']=='F'
    assert rows['/']['project_group']=='Q' and not rows['^']['project_group']
    assert rows['?']['policy']=='learning_beat_excluded_by_project_beat_set'
    for s in ('B','n','r'):assert rows[s]['policy']=='recognized_beat_unmapped_error_if_exported'
    for s in ('+','~','|','x'):assert not rows[s]['project_group']


def test_test_guard_before_file_io():
    index=SimpleNamespace(by_record={'999':SimpleNamespace(patient_group='test')})
    with pytest.raises(PermissionError):require_record(index,{'groups':{'train':['tr'],'validation':['v']}},'999')


def example(i,s='A',t='S',p='N',score=.5,record='201'):
    return dict(source_index=i,original_symbol=s,true_class=t,predicted_class=p,score=score,
        record_id=record,annotation_position=1000+i,partition='validation',patient_group='g'+record)


def test_selection_is_deterministic_semantic_and_not_max_confidence():
    rows=[example(1,score=.99),example(2,score=.5,record='202'),example(3,score=.49,record='210'),
          example(4,s='a',score=.6),example(5,t='N',p='N')]
    a=select_examples(rows,'S_to_N');b=select_examples(list(reversed(rows)),'S_to_N')
    assert a==b and a[0]['source_index']!=1
    assert all(r['true_class']=='S' and r['predicted_class']=='N' for r in a)
    assert {'A','a'}<=set(r['original_symbol'] for r in a)
    assert select_examples(rows,'F_correct')==[]


@pytest.fixture
def annotation_case():
    positions=np.array([50,100,200,600,950],dtype=np.int64);symbols=np.array(['N','N','A','V','N'])
    signal=np.arange(2000,dtype=np.float32).reshape(1000,2)
    keep=np.array([1,2,3]);X=np.array([signal[p-90:p+162].T for p in positions[keep]])
    arrays=dict(X=X,annotation_indices=keep,annotation_positions=positions[keep],original_symbols=symbols[keep],
                channel_names=np.array(['MLII','V1']),y=np.array([0,1,2]),record_ids=np.array(['201']*3))
    header=SimpleNamespace(sig_name=['MLII','V1'],fs=360,sig_len=1000)
    ann=SimpleNamespace(sample=positions,symbol=symbols)
    config=dict(before_samples=90,after_samples=162,filtering=dict(highpass_hz=.5,lowpass_hz=45.,notch_hz=None,order=4,quality_factor=30.),
        normalization=dict(scope='recording',sample_count=1000,method='zscore',channel_names=['MLII','V1'],source_ids=['201']))
    return arrays,ann,header,config


def test_original_annotated_window_alignment_and_overlap(annotation_case):
    arrays,ann,header,config=annotation_case
    before=arrays['X'].copy();r=verify_annotations(arrays,ann,header,config)
    assert r['exported_beats']==3 and r['skipped_boundary']==2 and r['overlap_pairs']==1
    assert arrays['X'][0,0,90]==200
    np.testing.assert_array_equal(before,arrays['X'])


@pytest.mark.parametrize('change',['position','symbol','channel','overlap','filter','normalization'])
def test_reproducible_consistency_errors_rejected(annotation_case,change):
    a,ann,h,c=annotation_case
    if change=='position':a['annotation_positions'][0]+=1
    if change=='symbol':a['original_symbols'][0]='V'
    if change=='channel':h.sig_name.reverse()
    if change=='overlap':a['X'][0,0,150]+=100
    if change=='filter':c['filtering']['highpass_hz']=1.
    if change=='normalization':c['normalization']['scope']='segment'
    with pytest.raises(ValueError):verify_annotations(a,ann,h,c)


def test_plot_contains_actual_labels_and_no_overwrite(tmp_path):
    arrays={'X':np.ones((1,2,252),dtype=np.float32),'annotation_positions':np.array([1001]),
        'original_symbols':np.array(['A']),'y':np.array([1]),'channel_names':np.array(['MLII','V1'])}
    row=example(1);p={'A':{1:row},'B':{1:row}}
    path=tmp_path/'example.png'
    draw_examples(path,[row],{'201':arrays},p,'Processed ECG validation error')
    assert path.is_file() and path.stat().st_size>10000
    with pytest.raises(FileExistsError):draw_examples(path,[row],{'201':arrays},p,'No overwrite')
    arrays['y'][0]=2
    with pytest.raises(ValueError,match='Plot class'):draw_examples(tmp_path/'wrong.png',[row],{'201':arrays},p,'Wrong label')
