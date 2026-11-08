"""New audit parsers only: no changes to heartbeat tests or implementations."""
import numpy as np
import pandas as pd
import pytest
from src.rhythm_audit import parse_rhythm_note,rhythm_intervals,parse_scp_codes,summarize_ptb_metadata


@pytest.mark.parametrize('note,expected',[('(AFIB\x00','AFIB'),(' (N ','N'),('(SBR)','SBR'),('(VT','VT')])
def test_documented_rhythm_notes(note,expected):
    assert parse_rhythm_note(note)==(expected,'documented_rhythm')


@pytest.mark.parametrize('note',['','N','AFIB','(N extra','(N\x00junk'])
def test_malformed_tags_do_not_become_normal(note):
    assert parse_rhythm_note(note)[0] is None


def test_unknown_tag_not_silently_normal():
    assert parse_rhythm_note('(STACH')==(None,'undocumented_rhythm_tag')


def intervals(samples,symbols,notes,**kwargs):
    return rhythm_intervals(samples,symbols,notes,sampling_frequency=10,sample_count=100,record_id='test',**kwargs)


def test_prefix_unknown_and_state_until_next_marker():
    result=intervals([10,20,50],['+','N','+'],['(N','(AFIB','(AFIB'])
    assert [(r['start_sample'],r['stop_sample'],r['label']) for r in result['intervals']]==[(0,10,None),(10,50,'N'),(50,100,'AFIB')]
    assert result['intervals'][-1]['right_censored']
    assert result['intervals'][1]['initial_state']
    assert len(result['ignored_rhythm_like_nonplus_notes'])==1
    assert sum(r['duration_samples'] for r in result['intervals'])==100
    assert sum(t['confirmed_label_change'] for t in result['transitions'])==1


def test_repeated_state_is_not_an_extra_episode():
    result=intervals([0,20,50],['+']*3,['(N','(N','(AFIB'])
    assert len(result['markers'])==3 and len(result['intervals'])==3
    assert len(result['episodes'])==2
    assert result['episodes'][0]['duration_seconds']==5
    assert result['episodes'][0]['source_annotation_indices']==[0,1]


def test_same_sample_conflict_is_ambiguous_not_last_wins():
    result=intervals([0,10,10,30],['+']*4,['(N','(AFIB','(VT','(N'])
    assert result['intervals'][1]['label'] is None
    assert result['intervals'][1]['status']=='conflicting_same_sample_markers'
    assert result['intervals'][1]['source_annotation_indices']==[1,2]
    assert result['intervals'][1]['duration_samples']==20


def test_same_sample_identical_tags_are_one_state():
    result=intervals([0,0],['+','+'],['(AFIB','(AFIB\x00'])
    assert len(result['episodes'])==1
    assert result['episodes'][0]['label']=='AFIB'


def test_malformed_marker_interrupts_previous_state():
    result=intervals([0,30,60],['+']*3,['(N','','(AFIB'])
    assert result['intervals'][1]['label'] is None
    assert result['intervals'][1]['duration_seconds']==3


def test_missing_rhythm_markers_leave_entire_record_unknown():
    result=intervals([10],['N'],[''])
    assert len(result['intervals'])==1
    assert result['intervals'][0]['label'] is None
    assert result['intervals'][0]['status']=='no_rhythm_annotations'
    assert result['intervals'][0]['duration_samples']==100


@pytest.mark.parametrize('samples,symbols,notes',[([-1],['+'],['(N']),([100],['+'],['(N']),
    ([1.5],['+'],['(N']),([20,10],['+','+'],['(N','(AFIB']),([0],[],[])])
def test_invalid_annotation_coordinates(samples,symbols,notes):
    with pytest.raises(ValueError):
        intervals(samples,symbols,notes)


def test_empty_annotation_arrays_and_half_open_boundaries():
    result=intervals([],[],[])
    assert result['intervals'][0]['start_seconds']==0
    assert result['intervals'][0]['stop_seconds']==10
    result=intervals([0,99],['+','+'],['(N','(VT'])
    assert result['episodes'][-1]['duration_samples']==1
    assert result['episodes'][-1]['duration_seconds']==0.1


def test_zero_scp_likelihood_is_a_present_code():
    assert parse_scp_codes("{'SR': 0.0, 'NORM': 100.0}")=={'SR':0.,'NORM':100.}


@pytest.mark.parametrize('text',["__import__('os').system('false')",'[]',"{'AFIB':-1}",
                                "{'AFIB':101}","{'SR':True}",'not a dictionary'])
def test_invalid_or_executable_scp_metadata_rejected(text):
    with pytest.raises(ValueError):
        parse_scp_codes(text)


def test_metadata_summary_on_small_test_fixture_only(tmp_path):
    # These are parser fixtures, NOT counts attributed to the actual PTB-XL dataset.
    database=tmp_path/'database.csv'
    definitions=tmp_path/'definitions.csv'
    pd.DataFrame({'ecg_id':[1,2,3],'patient_id':[10,10,20],
        'scp_codes':["{'SR':0,'NORM':100}","{'AFIB':0,'AFLT':100}","{'AFIB':100}"],
        'strat_fold':[1,1,10]}).to_csv(database,index=False)
    pd.DataFrame({'description':['sinus rhythm','atrial fibrillation','atrial flutter','normal ECG'],
                  'rhythm':[1,1,1,0]},index=['SR','AFIB','AFLT','NORM']).to_csv(definitions)
    summary=summarize_ptb_metadata(database,definitions)
    assert summary['record_count']==3 and summary['patient_count']==2
    assert summary['multiple_rhythm_code_records']==1
    assert summary['patients_across_multiple_folds']==0
    rows={r['code']:r for r in summary['rhythm_counts']}
    assert rows['AFIB']['record_count']==2 and rows['AFIB']['patient_count']==2
    assert rows['SR']['record_count']==1
    assert 'NORM' not in rows
    original=pd.read_csv(database)
    original.loc[1,'strat_fold']=2
    original.to_csv(database,index=False)
    assert summarize_ptb_metadata(database,definitions)['patients_across_multiple_folds']==1
    original.loc[1,'patient_id']=np.nan
    original.to_csv(database,index=False)
    with pytest.raises(ValueError):
        summarize_ptb_metadata(database,definitions)


def test_capacity_counts_arithmetic_only_without_extracting_signals():
    from src.rhythm_audit import rhythm_duration_capacity
    result=intervals([0,40,60],['+']*3,['(AFIB','(N','(AFIB'])
    audit={'records':{'test':{**result,'sampling_frequency':10,'patient_group':'test_group'}}}
    capacity={r['rhythm_code']:r for r in rhythm_duration_capacity(audit,duration_seconds=3)}
    assert capacity['AFIB']['theoretical_nonoverlapping_slots']==2
    assert capacity['AFIB']['qualifying_patient_groups']==1
    assert capacity['N']['theoretical_nonoverlapping_slots']==0
    assert capacity['N']['shorter_episodes']==1
    assert capacity['VT']['theoretical_nonoverlapping_slots']==0
    with pytest.raises(ValueError):
        rhythm_duration_capacity(audit,duration_seconds=0)
