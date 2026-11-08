"""Focused metadata parser/count tests; small fixtures are not dataset statistics."""
import copy
import json
import numpy as np
import pandas as pd
import pytest
from src.ptbxl_metadata import (parse_statement_dict,validate_metadata_frames,load_ptb_metadata,
    audit_metadata,propose_exclusive_cohort,MetadataValidationError,CORE_TARGETS)


@pytest.fixture
def frames():
    codes=['SR','AFIB','STACH','SBRAD','AFLT','SARRH','PACE','SVARR','SVTAC','PSVT','BIGU','TRIGU','NORM']
    definitions=pd.DataFrame({'description':[f'description {code}' for code in codes],
        'diagnostic':[0]*12+[1],'form':[0]*13,'rhythm':[1]*12+[0]},index=codes)
    statements=["{'SR':0,'NORM':100}","{'AFIB':0,'AFLT':100}","{'STACH':100}",
                "{'SBRAD':0}","{'NORM':100}","{'PACE':100}"]
    records=pd.DataFrame({'ecg_id':range(1,7),'patient_id':[10.,10.,20.,30.,40.,50.],
        'scp_codes':statements,'strat_fold':[1,1,9,10,2,3],
        'filename_lr':[f'records100/00000/{i:05d}_lr' for i in range(1,7)],
        'filename_hr':[f'records500/00000/{i:05d}_hr' for i in range(1,7)]})
    return records,definitions


def test_safe_parsing_and_original_likelihoods():
    assert parse_statement_dict("{'AFIB':0.0,'AFLT':100.0}")=={'AFIB':0.,'AFLT':100.}
    assert parse_statement_dict({})=={}


@pytest.mark.parametrize('value',["__import__('os').system('false')","{'SR':0,'SR':100}",
    "{'AFIB': -1}","{'AFIB': 101}","{'SR': True}","{'SR':'unknown'}",'[]',np.nan,'',"{1:0}"])
def test_malformed_or_duplicate_scp_rejected(value):
    with pytest.raises(MetadataValidationError):
        parse_statement_dict(value)


def test_counts_patients_folds_cooccurrences_and_no_mutation(frames):
    records,definitions=frames
    records_before=records.copy(deep=True)
    definitions_before=definitions.copy(deep=True)
    data=validate_metadata_frames(records,definitions)
    result=audit_metadata(data)
    assert result['record_count']==6 and result['patient_count']==5
    assert result['records_without_rhythm']==1 and result['records_with_multiple_rhythms']==1
    rows={r['code']:r for r in result['rhythm_counts']}
    assert rows['AFIB']['record_count']==rows['AFIB']['patient_count']==1
    assert rows['AFIB']['zero_likelihood_records']==1
    assert rows['AFIB']['records_by_fold']['1']==1
    assert rows['AFIB']['likelihood_counts']=={'0':1}
    assert rows['PSVT']['record_count']==0
    pair=next(r for r in result['cooccurrence'] if r['code_a']=='AFIB' and r['code_b']=='AFLT')
    assert pair['record_count']==pair['patient_count']==1
    reverse=next(r for r in result['cooccurrence'] if r['code_a']=='AFLT' and r['code_b']=='AFIB')
    assert reverse['record_count']==pair['record_count']
    assert result['NORM_without_SR']==1
    assert result['patient_fold_conflict_count']==0
    pd.testing.assert_frame_equal(records,records_before)
    pd.testing.assert_frame_equal(definitions,definitions_before)
    assert audit_metadata(data)==result


def test_explicit_proposal_counts_before_after_and_reasons(frames):
    data=validate_metadata_frames(*frames)
    result=propose_exclusive_cohort(data)
    assert result['targets']==list(CORE_TARGETS)
    assert result['proposed_record_count']==result['proposed_patient_count']==3
    assert result['mutually_exclusive_exclusion_reasons']=={'proposed_eligible':3,
        'multiple_rhythm_statements':1,'no_rhythm_statement':1,'single_rhythm_outside_proposed_targets':1}
    assert result['per_class']['AFIB']['records_before']==1
    assert result['per_class']['AFIB']['records_after']==0
    assert result['per_class']['STACH']['records_after_by_fold']['9']==1
    assert len(result['record_audit'])==6
    assert data.statements[1]=={'AFIB':0,'AFLT':100}
    assert result['record_audit'][1]['proposed_label'] is None


def test_patient_fold_conflict_reported_without_reassignment(frames):
    records,definitions=frames
    records.loc[1,'strat_fold']=5
    data=validate_metadata_frames(records,definitions)
    result=audit_metadata(data)
    assert result['patient_fold_conflict_count']==1
    assert result['patient_fold_conflicts']=={'10':[1,5]}
    assert data.folds.tolist()==[1,5,9,10,2,3]


@pytest.mark.parametrize('column',['ecg_id','patient_id','scp_codes','strat_fold','filename_lr','filename_hr'])
def test_required_columns(frames,column):
    records,definitions=frames
    with pytest.raises(MetadataValidationError):
        validate_metadata_frames(records.drop(columns=column),definitions)


@pytest.mark.parametrize('column,value',[('ecg_id',np.nan),('ecg_id',1.5),('patient_id',np.nan),
    ('patient_id',0),('patient_id','bad'),('strat_fold',np.nan),('strat_fold',11),('strat_fold',2.5),
    ('filename_lr',np.nan),('filename_hr','../00001_hr'),
    ('filename_lr','records500/00000/00001_lr'),('filename_hr','records500/01000/00001_hr'),
    ('filename_lr','records100/00000/00002_lr'),('scp_codes',np.nan),('scp_codes',"{'UNKNOWN':0}")])
def test_invalid_required_values(frames,column,value):
    records,definitions=frames
    records[column]=records[column].astype(object)
    records.loc[0,column]=value
    with pytest.raises(MetadataValidationError):
        validate_metadata_frames(records,definitions)


def test_duplicate_ecg_ids_and_definition_codes(frames):
    records,definitions=frames
    records.loc[1,'ecg_id']=1
    with pytest.raises(MetadataValidationError,match='Duplicate'):
        validate_metadata_frames(records,definitions)
    records.loc[1,'ecg_id']=2
    duplicate=pd.concat([definitions,definitions.iloc[:1]])
    with pytest.raises(MetadataValidationError):
        validate_metadata_frames(records,duplicate)


@pytest.mark.parametrize('column,value',[('description',np.nan),('description',''),('rhythm',2),('form','bad')])
def test_invalid_definition_values(frames,column,value):
    records,definitions=frames
    definitions[column]=definitions[column].astype(object)
    definitions.loc['SR',column]=value
    with pytest.raises(MetadataValidationError):
        validate_metadata_frames(records,definitions)


def test_optional_nulls_are_reported_not_imputed_or_excluded(frames):
    records,definitions=frames
    records['baseline_drift']=[np.nan]*6
    result=audit_metadata(validate_metadata_frames(records,definitions))
    assert result['missing_values_by_column']['baseline_drift']==6
    assert result['proposal']['proposed_record_count']==3
    assert 'validated_by_human' in result['optional_quality_columns_missing']


def test_empty_input_and_invalid_target_choices(frames):
    records,definitions=frames
    data=validate_metadata_frames(records.iloc[:0],definitions)
    result=audit_metadata(data)
    assert result['record_count']==result['patient_count']==0
    assert result['proposal']['proposed_record_count']==0
    for targets in [[],['SR','SR'],['NORM'],['UNKNOWN']]:
        with pytest.raises(MetadataValidationError):
            propose_exclusive_cohort(data,targets)


def test_local_csv_loading_and_missing_file_error(frames,tmp_path):
    with pytest.raises(FileNotFoundError,match='official PTB-XL 1.0.3'):
        load_ptb_metadata(tmp_path)
    records,definitions=frames
    records.to_csv(tmp_path/'ptbxl_database.csv',index=False)
    definitions.to_csv(tmp_path/'scp_statements.csv')
    data=load_ptb_metadata(tmp_path)
    assert len(data.ecg_ids)==6
    assert len(data.provenance['files']['ptbxl_database.csv']['sha256'])==64
    assert data.statements[1]['AFIB']==0


def test_actual_local_metadata_integration():
    data=load_ptb_metadata()
    result=audit_metadata(data)
    assert result['record_count']==21799 and result['patient_count']==18869
    assert result['definition_count']==71
    assert result['matches_published_1_0_3_cardinalities']
    assert result['patient_fold_conflict_count']==0
    assert result['records_without_rhythm']==769
    assert result['records_with_multiple_rhythms']==143
    expected={'SR':(16748,15035),'AFIB':(1514,1245),'STACH':(826,782),'SBRAD':(637,616),
              'AFLT':(73,56),'SARRH':(772,757),'PACE':(294,237),'SVARR':(157,148),
              'BIGU':(82,79),'SVTAC':(27,27),'PSVT':(24,24),'TRIGU':(20,20)}
    assert {r['code']:(r['record_count'],r['patient_count']) for r in result['rhythm_counts']}==expected
    for row in result['rhythm_counts']:
        assert sum(row['records_by_fold'].values())==row['record_count']
        assert sum(row['patients_by_fold'].values())==row['patient_count']
