"""Explicit mapping, one-to-one metadata retention, distributions and local integration."""
from dataclasses import replace
import numpy as np
import pytest
from src.segmentation import segment_heartbeats, segment_recording
from src.data_loader import load_recording
from src.label_mapping import (CLASS_NAMES, CLASS_INDICES, DEFAULT_MAPPING, map_symbol,
    map_segments, class_distribution, summarize_recordings, validate_mapping)

EXPECTED = {'N':('N',0),'L':('N',0),'R':('N',0),'e':('N',0),'j':('N',0),
            'A':('S',1),'a':('S',1),'J':('S',1),'S':('S',1),
            'V':('V',2),'E':('V',2),'F':('F',3),'/':('Q',4),'f':('Q',4),'Q':('Q',4)}


def segments(symbols):
    return segment_heartbeats(np.zeros((2000,2)),360,np.arange(len(symbols))+300,symbols,
        record_id='source',channel_names=('MLII','V5'),units=('dimensionless',)*2,
        patient_group_id='provided_group',annotation_aux_notes=[f'note{i}' for i in range(len(symbols))])


@pytest.mark.parametrize('symbol,expected',EXPECTED.items())
def test_exact_mapping_and_index(symbol,expected):
    assert map_symbol(symbol) == expected
    assert map_symbol(symbol) == map_symbol(symbol)


def test_complete_default_map_and_fixed_class_indices():
    assert dict(CLASS_INDICES) == {'N':0,'S':1,'V':2,'F':3,'Q':4}
    assert dict(DEFAULT_MAPPING) == {s:v[0] for s,v in EXPECTED.items()}
    result = map_segments(segments(list(EXPECTED)))
    np.testing.assert_array_equal(result.y,[v[1] for v in EXPECTED.values()])
    assert result.y.dtype == np.int64
    assert result.class_names == CLASS_NAMES


@pytest.mark.parametrize('symbol',['B','n','r','?','+','~','|','x','Z',''])
def test_no_fallback_for_unknown_or_excluded(symbol):
    with pytest.raises(ValueError,match='Unmapped'):
        map_symbol(symbol)


def test_mark_retains_unmapped_with_explicit_sentinel():
    source=segments(['N','B','n','r','V'])
    with pytest.raises(ValueError,match='Unmapped'):
        map_segments(source)
    result=map_segments(source,unknown_policy='mark')
    assert result.segments is source
    assert result.X is source.X
    assert result.y.tolist() == [0,-1,-1,-1,2]
    assert result.mapped_classes.tolist() == ['N','UNMAPPED','UNMAPPED','UNMAPPED','V']
    assert result.mapped_mask.tolist() == [True,False,False,False,True]
    data=class_distribution(result)
    assert data['unmapped_symbol_counts'] == {'B':1,'n':1,'r':1}
    assert data['class_percentages_of_mapped']['N'] == 50
    assert data['class_percentages_of_segments']['N'] == 20
    assert data['segment_count'] == 5


def test_custom_map_is_validated_detached_and_deterministic():
    custom={'r':'V','B':'N','n':'N'}
    result=map_segments(segments(['B','n','r']),custom)
    assert result.y.tolist()==[0,0,2]
    assert list(result.mapping)==['B','n','r']
    custom['B']='Q'
    assert result.mapping['B']=='N'
    assert 'B' not in DEFAULT_MAPPING
    with pytest.raises(ValueError):
        map_symbol('N',{'V':'V'})  # Custom map replaces defaults.


@pytest.mark.parametrize('mapping',[{'N':'bad'},{'+':'N'},{'?':'Q'},{'bad':'N'},[],{'N':0}])
def test_invalid_mapping(mapping):
    with pytest.raises(ValueError):
        validate_mapping(mapping)


def test_metadata_preserved_in_original_order_without_mutation():
    source=segments(['V','N','A','F','Q'])
    original=source.X.copy()
    result=map_segments(source)
    assert result.segments is source
    np.testing.assert_array_equal(source.X,original)
    assert result.segments.annotation_symbols.tolist()==['V','N','A','F','Q']
    assert result.segments.record_ids.tolist()==['source']*5
    assert result.segments.annotation_positions.tolist()==list(range(300,305))
    assert result.segments.annotation_indices.tolist()==list(range(5))
    assert result.segments.annotation_aux_notes.tolist()==[f'note{i}' for i in range(5)]
    assert result.segments.patient_group_ids.tolist()==['provided_group']*5
    assert result.segments.channel_names==('MLII','V5')
    assert result.segments.sampling_frequency==360
    assert result.segments.configuration==source.configuration


def test_unresolved_patient_identity_is_not_invented():
    source=segment_heartbeats(np.zeros(1000),360,[300],['N'],record_id='100')
    result=map_segments(source)
    assert result.segments.patient_group_ids.tolist()==['']
    assert class_distribution(result)['patient_groups_unresolved_count']==1


def test_counts_percentages_and_original_symbols():
    result=map_segments(segments(['N','L','A','V','F','/']))
    data=class_distribution(result)
    assert data['class_counts']=={'N':2,'S':1,'V':1,'F':1,'Q':1}
    assert sum(data['class_counts'].values())==len(result.X)
    assert sum(data['class_percentages_of_mapped'].values())==pytest.approx(100)
    assert data['class_percentages_of_mapped']['N']==pytest.approx(100/3)
    assert data['original_segment_symbol_counts']=={'/':1,'A':1,'F':1,'L':1,'N':1,'V':1}


def test_empty_input():
    result=map_segments(segments([]))
    assert result.y.shape==(0,)
    assert result.X.shape==(0,2,252)
    data=class_distribution(result)
    assert data['mapped_count']==data['unmapped_count']==0
    assert all(v==0 for v in data['class_counts'].values())
    assert all(v==0 for v in data['class_percentages_of_mapped'].values())


def test_all_unmapped():
    data=class_distribution(map_segments(segments(['B']),unknown_policy='mark'))
    assert data['unmapped_count']==1
    assert sum(data['class_counts'].values())==0
    assert sum(data['class_percentages_of_mapped'].values())==0


def test_bad_policy_or_metadata():
    source=segments(['N'])
    with pytest.raises(ValueError):
        map_segments(source,unknown_policy='drop')
    with pytest.raises(ValueError):
        map_segments(replace(source,annotation_positions=np.array([],dtype=int)))


def test_summary_default_all_records_is_streamed_and_totals_consistent(monkeypatch):
    import src.label_mapping as module
    calls=[]
    monkeypatch.setattr(module,'list_recordings',lambda root:['100','101'])
    monkeypatch.setattr(module,'load_recording',lambda rid,root:rid)
    def segment(rid):
        calls.append(rid)
        return segments(['N','V'] if rid=='100' else ['A','B'])
    # Record annotations are inspected separately from the segment objects.
    from types import SimpleNamespace
    monkeypatch.setattr(module,'load_recording',lambda rid,root:SimpleNamespace(record_id=rid,annotation_symbols=['N']))
    monkeypatch.setattr(module,'segment_recording',lambda record:segment(record.record_id))
    report=summarize_recordings('unused')
    assert calls==['100','101']
    assert report['aggregate']['class_counts']=={'N':1,'S':1,'V':1,'F':0,'Q':0}
    assert report['aggregate']['unmapped_symbol_counts']=={'B':1}
    assert report['aggregate']['segment_count']==4
    with pytest.raises(ValueError):
        summarize_recordings('unused',['100','100'])
    empty=summarize_recordings('unused',[])
    assert empty['aggregate']['segment_count']==0


@pytest.mark.parametrize('rid,total',[('100',2271),('101',1864),('200',2600)])
def test_actual_segmentation_integration(rid,total):
    source=segment_recording(load_recording(rid))
    result=map_segments(source)
    data=class_distribution(result)
    assert len(result.y)==total==len(source.X)
    assert data['mapped_count']==total
    assert data['unmapped_count']==0
    assert sum(data['class_counts'].values())==total
    assert result.segments.annotation_symbols is source.annotation_symbols
    np.testing.assert_array_equal(result.segments.annotation_positions,source.annotation_positions)
