"""Protocol tests use explicit synthetic NPZ fixtures; real-cohort tests use local exports."""
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import pytest
from src.ptbxl_protocol import (PTBXLExportIndex,file_sha256,official_partitions,
                               training_class_counts,training_class_weights)
from src.ptbxl_model_config import RhythmCNNConfig


@pytest.fixture
def fixture_export(tmp_path):
    root=tmp_path/'dataset';(root/'recordings').mkdir(parents=True)
    codes=['SR','SR','SR','AFIB','STACH','SBRAD','SBRAD','SR']
    folds=[1,2,3,4,5,6,9,10];classes=['SR','AFIB','STACH','SBRAD']
    rows=[]
    for i,(code,fold) in enumerate(zip(codes,folds)):
        eid=90001+i;pid=5001+i;y=classes.index(code)
        identity={'ecg_id':eid,'patient_id':pid,'strat_fold':fold,'label':code,'original_scp_statements':{code:0.}}
        path=root/f'recordings/{eid:05d}_rhythm.npz'
        np.savez_compressed(path,X=np.zeros((12,1000),dtype=np.float32),y=np.array(y,dtype=np.int64),
            ecg_id=np.array(eid,dtype=np.int64),patient_id=np.array(pid,dtype=np.int64),strat_fold=np.array(fold,dtype=np.int64),
            rhythm_code=np.array(code),original_scp_codes=np.array(json.dumps({code:0.})),original_metadata=np.array(json.dumps(identity)),
            channel_names=np.array(['I','II','III','AVR','AVL','AVF','V1','V2','V3','V4','V5','V6']),
            class_names=np.array(classes),sampling_frequency=np.array(100.),preprocessing_configuration=np.array('{}'),source_hashes=np.array('{}'))
        rows.append({'ecg_id':eid,'patient_id':pid,'strat_fold':fold,'label':code,'file':str(path.relative_to(root)),
                     'sha256':file_sha256(path),'status':'exported'})
    manifest={'status':'complete','records':rows,'selected_records':8,'successful_records':8,
              'successful_patients':8,'failed_records':0,'not_attempted_records':0}
    path=root/'manifest.json';path.write_text(json.dumps(manifest))
    return root,path,manifest


def fixture_index(fixture_export):
    root,path,_=fixture_export
    return PTBXLExportIndex(root,path,require_full_cohort=False,verify_source_metadata=False)


def test_deterministic_index_and_original_fold_selection(fixture_export):
    index=fixture_index(fixture_export)
    assert len(index)==8
    assert index.partitions.train==tuple(range(6))
    assert index.partitions.validation==(6,) and index.partitions.test==(7,)
    assert official_partitions(index)==index.partitions
    assert [e.ecg_id for e in index.entries]==list(range(90001,90009))
    assert [e.label for e in index.entries]==[0,0,0,1,2,3,3,0]


def test_record_reader_shape_types_and_original_ids(fixture_export):
    index=fixture_index(fixture_export);arrays=index.load_numpy(3)
    assert arrays['X'].shape==(12,1000) and arrays['X'].dtype==np.float32
    assert arrays['y'].shape==() and arrays['y'].dtype==np.int64 and int(arrays['y'])==1
    assert int(arrays['ecg_id'])==90004 and int(arrays['patient_id'])==5004 and int(arrays['strat_fold'])==4


def test_training_only_counts_and_balanced_weights(fixture_export):
    index=fixture_index(fixture_export)
    counts=training_class_counts(index,index.partitions.train)
    np.testing.assert_array_equal(counts,[3,1,1,1])
    weights=training_class_weights(index,index.partitions.train)
    np.testing.assert_allclose(weights,[.5,1.5,1.5,1.5])
    assert weights.dtype==np.float32
    assert np.isfinite(weights).all()


@pytest.mark.parametrize('indices',[None,[],[True],[0,0],[-1],[8],[0.5],[6],[7],list(range(8))])
def test_weights_reject_unsafe_or_nontraining_indices(fixture_export,indices):
    index=fixture_index(fixture_export)
    with pytest.raises(ValueError):training_class_weights(index,indices)


def test_missing_training_class_is_explicit(fixture_export):
    with pytest.raises(ValueError,match='All four classes'):
        training_class_weights(fixture_index(fixture_export),(0,1,2,3))


def test_patient_overlap_rejected_even_before_tensor_loading(fixture_export):
    root,path,manifest=fixture_export
    manifest['records'][6]['patient_id']=5001;manifest['successful_patients']=7
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='Patient.*official fold'):fixture_index(fixture_export)


def test_incomplete_run_missing_file_and_duplicate_not_silently_skipped(fixture_export):
    root,path,manifest=fixture_export
    manifest['status']='incomplete';path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='complete'):fixture_index(fixture_export)
    manifest['status']='complete';manifest['records'][1]=manifest['records'][0]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='Duplicate'):fixture_index(fixture_export)


def test_missing_record_clear_error(fixture_export):
    root,path,manifest=fixture_export
    (root/manifest['records'][0]['file']).unlink()
    with pytest.raises(FileNotFoundError,match='ecg_id=90001'):fixture_index(fixture_export)


def test_changed_record_checksum_and_metadata_are_rejected(fixture_export):
    index=fixture_index(fixture_export);entry=index.entries[0];path=index.root/entry.file
    path.write_bytes(b'invalid npz')
    with pytest.raises(ValueError,match='ecg_id=90001.*SHA256'):index.load_numpy(0)
    with pytest.raises(ValueError,match='ecg_id=90001'):index.load_numpy(0,verify_checksum=False)


@pytest.mark.parametrize('change',['shape','dtype','nan','label','patient','fold'])
def test_malformed_arrays_are_never_coerced_or_skipped(fixture_export,change):
    index=fixture_index(fixture_export);entry=index.entries[0];path=index.root/entry.file
    with np.load(path,allow_pickle=False) as archive:arrays={k:archive[k] for k in archive.files}
    if change=='shape':arrays['X']=arrays['X'][:,:999]
    if change=='dtype':arrays['X']=arrays['X'].astype(np.float64)
    if change=='nan':arrays['X'][0,0]=np.nan
    if change=='label':arrays['y']=np.array(4,dtype=np.int64)
    if change=='patient':arrays['patient_id']=np.array(999,dtype=np.int64)
    if change=='fold':arrays['strat_fold']=np.array(10,dtype=np.int64)
    np.savez_compressed(path,**arrays)
    index.entries=(replace(entry,sha256=file_sha256(path)),)+index.entries[1:]
    with pytest.raises(ValueError,match='ecg_id=90001'):index.load_numpy(0)


def test_full_cohort_guard_and_safe_path(fixture_export):
    root,path,manifest=fixture_export
    with pytest.raises(ValueError,match='Phase-12'):PTBXLExportIndex(root,path,verify_source_metadata=False)
    manifest['records'][0]['file']='../escape.npz';path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='path'):fixture_index(fixture_export)


def test_configured_model_parameter_arithmetic_and_contract():
    config=RhythmCNNConfig()
    assert config.expected_parameter_count()==74116
    assert config.to_dict()['num_classes']==4
    for kwargs in ({'input_channels':2},{'sequence_length':252},{'num_classes':5},{'dropout':float('nan')},{'dropout':1},{'kernels':(4,9,5)}):
        with pytest.raises(ValueError):RhythmCNNConfig(**kwargs)


def test_real_completed_cohort_patient_safety_and_training_counts():
    index=PTBXLExportIndex()
    assert len(index)==19605 and len({e.patient_id for e in index.entries})==17223
    groups=[set(getattr(index.partitions,p)) for p in ('train','validation','test')]
    assert set.union(*groups)==set(range(19605))
    for i in range(3):
        for j in range(i):
            assert not groups[i]&groups[j]
            assert not {index.entries[k].patient_id for k in groups[i]}&{index.entries[k].patient_id for k in groups[j]}
    counts=training_class_counts(index,index.partitions.train)
    assert counts.sum()==len(index.partitions.train) and (counts>0).all()
    # Confirm all four real class labels using training records, never final test data.
    for label in range(4):
        pos=next(i for i in index.partitions.train if index.entries[i].label==label)
        data=index.load_numpy(pos)
        assert int(data['y'])==label and int(data['strat_fold']) in range(1,9)
