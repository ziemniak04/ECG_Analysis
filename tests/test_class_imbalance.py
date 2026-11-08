"""Descriptive audit and isolated training-only utility behavior."""
from dataclasses import replace
import numpy as np
import pytest
from src.segmentation import segment_heartbeats, segment_recording
from src.label_mapping import CLASS_NAMES, map_segments
from src.data_loader import load_recording
from src.class_imbalance import (describe_counts, analyze_labeled, analyze_database,
    make_training_subset, training_class_weights, training_sampling_indices)


def labeled(symbols=('N','N','N','V','F','S','Q','N')):
    segments=segment_heartbeats(np.zeros((1000,2)),360,np.arange(len(symbols))+300,symbols,
        record_id='source',channel_names=('MLII','V5'),units=('dimensionless',)*2,patient_group_id='known_group')
    return map_segments(segments,unknown_policy='mark')


def test_known_counts_percentages_and_metadata():
    data=labeled()
    original=data.X.copy()
    report=analyze_labeled(data,rare_count_threshold=2)
    assert report['class_counts']==dict(zip(CLASS_NAMES,[4,1,1,1,1]))
    assert report['class_percentages']['N']==50
    assert sum(report['class_percentages'].values())==100
    assert report['majority_classes']==['N']
    assert report['minority_nonempty_classes']==['S','V','F','Q']
    assert report['majority_to_minority_nonempty_ratio']==4
    assert report['rare_classes']==['S','V','F','Q']
    assert report['record_patient_groups']==[{'record_id':'source','patient_group_id':'known_group'}]
    np.testing.assert_array_equal(data.X,original)
    assert data.segments.annotation_positions.tolist()==list(range(300,308))
    assert analyze_labeled(data,rare_count_threshold=2)==report


def test_missing_and_unmapped_classes():
    report=analyze_labeled(labeled(['N','N','B']))
    assert report['mapped_count']==2 and report['unmapped_count']==1
    assert report['zero_classes']==['S','V','F','Q']
    assert report['majority_classes']==report['minority_nonempty_classes']==['N']
    assert report['majority_to_minority_nonempty_ratio']==1
    assert report['class_percentages']['N']==100
    assert report['unmapped_symbol_counts']=={'B':1}


def test_empty_input():
    report=analyze_labeled(labeled([]))
    assert report['mapped_count']==0
    assert report['majority_to_minority_nonempty_ratio'] is None
    assert report['majority_classes']==report['minority_nonempty_classes']==[]
    assert report['zero_classes']==list(CLASS_NAMES)
    assert report['rare_classes']==[]
    assert sum(report['class_percentages'].values())==0


@pytest.mark.parametrize('counts,threshold',[
    ({'N':1},1000), (dict.fromkeys(CLASS_NAMES,-1),1000),
    (dict.fromkeys(CLASS_NAMES,1.5),1000),(dict.fromkeys(CLASS_NAMES,True),1000),
    (dict.fromkeys(CLASS_NAMES,0),0),(dict.fromkeys(CLASS_NAMES,0),True)])
def test_invalid_descriptive_inputs(counts,threshold):
    with pytest.raises(ValueError):
        describe_counts(counts,rare_count_threshold=threshold)


def test_inconsistent_labels_rejected():
    source=labeled()
    with pytest.raises(ValueError):
        analyze_labeled(replace(source,y=np.array([0])))
    with pytest.raises(ValueError):
        analyze_labeled(replace(source,y=np.full(len(source.y),5)))
    with pytest.raises(ValueError):
        analyze_labeled(replace(source,mapped_classes=np.full(len(source.y),'Q')))


def train():
    # Synthetic utility fixture, not the actual database analysis or a patient split.
    return make_training_subset(labeled(),[0,1,2,3,4],partition='train',patient_split_confirmed=True)


def test_training_weight_formula_absent_classes_and_source_metadata():
    subset=train()
    weights=training_class_weights(subset)
    assert weights=={0:pytest.approx(5/9),1:0.,2:pytest.approx(5/3),3:pytest.approx(5/3),4:0.}
    assert sum(weights[int(y)] for y in subset.y)==pytest.approx(5)
    assert subset.record_ids.tolist()==['source']*5
    assert subset.patient_group_ids.tolist()==['known_group']*5
    assert subset.annotation_positions.tolist()==list(range(300,305))
    assert not subset.y.flags.writeable


@pytest.mark.parametrize('strategy,count',[('oversample',3),('undersample',1)])
def test_sampling_indices_present_classes_only_deterministic_and_nonmutation(strategy,count):
    subset=train()
    before=subset.y.copy()
    indices=training_sampling_indices(subset,strategy=strategy,random_seed=42)
    np.testing.assert_array_equal(indices,training_sampling_indices(subset,strategy=strategy,random_seed=42))
    assert set(indices).issubset(set(subset.source_indices))
    labels=labeled().y[indices]
    assert np.bincount(labels,minlength=5).tolist()==[count,0,count,count,0]
    if strategy=='undersample':
        assert len(np.unique(indices))==len(indices)
    else:
        assert set(subset.source_indices).issubset(set(indices))
    np.testing.assert_array_equal(before,subset.y)


def test_sampling_returns_source_not_subset_offsets():
    subset=make_training_subset(labeled(),[1,2,3,4],partition='train',patient_split_confirmed=True)
    result=training_sampling_indices(subset,strategy='oversample',random_seed=0)
    assert set(result)=={1,2,3,4} and 0 not in result


@pytest.mark.parametrize('partition,confirmed',[('validation',True),('test',True),('train',False),('train',1)])
def test_only_explicit_confirmed_train_allowed(partition,confirmed):
    with pytest.raises(ValueError):
        make_training_subset(labeled(),[0,1],partition=partition,patient_split_confirmed=confirmed)


@pytest.mark.parametrize('indices',[[],[0,0],[-1],[8],[0.5],[True,False],list(range(8))])
def test_invalid_training_selections_and_full_dataset(indices):
    with pytest.raises(ValueError):
        make_training_subset(labeled(),indices,partition='train',patient_split_confirmed=True)


def test_unresolved_groups_and_unmapped_training_labels_rejected():
    data=labeled()
    unresolved=replace(data,segments=replace(data.segments,patient_group_ids=np.full(8,'')))
    with pytest.raises(ValueError):
        make_training_subset(unresolved,[0,1],partition='train',patient_split_confirmed=True)
    with pytest.raises(ValueError):
        make_training_subset(labeled(['N','B','V']),[0,1],partition='train',patient_split_confirmed=True)


@pytest.mark.parametrize('value',[None,np.array([0,0,1]),'train'])
def test_utilities_reject_raw_labels_or_missing_subset(value):
    with pytest.raises(ValueError):
        training_class_weights(value)
    with pytest.raises(ValueError):
        training_sampling_indices(value,strategy='oversample',random_seed=1)


@pytest.mark.parametrize('strategy,seed',[('smote',1),('oversample',None),('undersample',-1),('oversample',True)])
def test_invalid_sampling_configuration(strategy,seed):
    with pytest.raises(ValueError):
        training_sampling_indices(train(),strategy=strategy,random_seed=seed)


def test_empty_database_summary_and_duplicate_ids():
    report=analyze_database(record_ids=[])
    assert report['record_count']==0 and report['aggregate']['segment_count']==0
    with pytest.raises(ValueError):
        analyze_database(record_ids=['100','100'])


def test_streaming_aggregation_uses_each_record_once(monkeypatch):
    import src.class_imbalance as module
    from types import SimpleNamespace
    calls=[]
    monkeypatch.setattr(module,'list_recordings',lambda root:['a','b'])
    monkeypatch.setattr(module,'load_recording',lambda rid,root:SimpleNamespace(record_id=rid))
    def prepare(record):
        calls.append(record.record_id)
        return labeled(['N','V'] if record.record_id=='a' else ['F','F']).segments
    monkeypatch.setattr(module,'segment_recording',prepare)
    progress=[]
    result=analyze_database(progress=lambda rid,r:progress.append(rid))
    assert calls==progress==['a','b']
    assert result['aggregate']['class_counts']=={'N':1,'S':0,'V':1,'F':2,'Q':0}
    assert result['aggregate']['total_annotations']==4
    assert result['records_missing_classes']['a']==['S','F','Q']


@pytest.mark.parametrize('rid,expected',[('100',2271),('101',1864),('200',2600)])
def test_real_labeled_heartbeat_integration(rid,expected):
    data=map_segments(segment_recording(load_recording(rid)))
    report=analyze_labeled(data)
    assert report['mapped_count']==report['segment_count']==expected
    assert sum(report['class_counts'].values())==expected
    assert report['patient_groups_unresolved_count']==expected
    assert report['record_patient_groups']==[{'record_id':rid,'patient_group_id':None}]
