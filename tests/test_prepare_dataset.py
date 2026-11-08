"""Final NPZ/JSON handoff, original metadata and complete-database integrity."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
from src.data_loader import DEFAULT_DATA_DIR, locate_data_dir, load_recording
from src.segmentation import segment_recording
from src.label_mapping import map_segments, CLASS_NAMES
from src.prepare_dataset import (DEFAULT_OUTPUT_DIR, DEFAULT_PATIENT_FILE, PHASE9_COUNTS,
    PER_BEAT_FIELDS, export_dataset, verify_export, load_patient_groups,
    load_record_shard, validate_shard, source_hashes)


@pytest.fixture(scope='module')
def small_export(tmp_path_factory):
    directory=tmp_path_factory.mktemp('export_parent')/'record_100'
    metadata=export_dataset(output_dir=directory,record_ids=['100'])
    return directory,metadata


def test_documented_patient_groups_and_relationship_provenance():
    metadata=load_patient_groups()
    rows=metadata['recordings']
    assert len(rows)==48
    assert len({r['patient_group'] for r in rows.values()})==47
    assert rows['201']['patient_group']==rows['202']['patient_group']
    assert rows['201']['relationship_basis']=='explicit_same_subject_201_202'
    assert rows['100']['relationship_basis']=='distinct_by_documented_cardinality'
    assert all(r['real_patient_identity'] is None for r in rows.values())
    assert metadata['sources'][0]=='https://physionet.org/content/mitdb/1.0.0/'


def test_bad_patient_relationship_rejected(tmp_path):
    metadata=load_patient_groups()
    metadata['recordings']['202']['patient_group']='invented_different_group'
    path=tmp_path/'bad.json'
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        load_patient_groups(path)


def test_export_fields_shapes_dtypes_and_finiteness(small_export):
    directory,metadata=small_export
    shard=load_record_shard(directory/'recordings/100.npz')
    assert shard['X'].shape==(2271,2,252)
    assert shard['X'].dtype==np.float32
    assert shard['y'].dtype==np.int64
    assert np.isfinite(shard['X']).all()
    assert np.all((shard['y']>=0)&(shard['y']<5))
    assert all(len(shard[name])==2271 for name in PER_BEAT_FIELDS)
    assert shard['class_names'].tolist()==list(CLASS_NAMES)
    assert shard['channel_names'].tolist()==['MLII','V5']
    assert float(shard['sampling_frequency'])==360
    assert metadata['scope']=='explicit_subset'
    assert metadata['phase9_counts_match'] is None
    assert metadata['segment_count']==2271


def test_original_annotations_and_preprocessing_values_preserved(small_export):
    directory,_=small_export
    shard=load_record_shard(directory/'recordings/100.npz')
    raw=load_recording('100')
    expected=map_segments(segment_recording(raw,patient_group_id='mitdb_group_100'))
    np.testing.assert_array_equal(shard['X'],expected.X.astype(np.float32))
    np.testing.assert_array_equal(shard['y'],expected.y)
    np.testing.assert_array_equal(shard['annotation_positions'],expected.segments.annotation_positions)
    np.testing.assert_array_equal(shard['original_symbols'],expected.segments.annotation_symbols)
    np.testing.assert_array_equal(shard['annotation_indices'],expected.segments.annotation_indices)
    assert np.all(shard['record_ids']=='100')
    assert np.all(shard['patient_groups']=='mitdb_group_100')
    config=json.loads(str(shard['preprocessing_config']))
    assert config['normalization']['method']=='zscore'
    assert config['normalization']['scope']=='recording'
    assert config['normalization']['source_ids']==['100']
    assert config['filtering']=={'highpass_hz':0.5,'lowpass_hz':45.,'order':4,'notch_hz':None,'quality_factor':30.}
    assert config['before_samples']==90 and config['after_samples']==162
    assert config['mapping_version']=='mitdb-aami-style-v1'


def test_required_manifest_metadata_and_readback(small_export):
    directory,metadata=small_export
    required={'format_version','status','class_counts','class_names','mapping','mapping_version',
        'source_files_sha256','source_files_unchanged','package_versions','code_sha256',
        'channel_policy','normalization_scope','shards','patient_mapping_version','no_split','no_balancing'}
    assert required.issubset(metadata)
    assert metadata['no_split'] is True and metadata['no_balancing'] is True
    assert verify_export(directory,source_root=DEFAULT_DATA_DIR)['source_files_verified']
    with (directory/'dataset_metadata.json').open() as source:
        assert json.load(source)==metadata


def test_sources_unchanged_and_hashes_match(small_export):
    _,metadata=small_export
    assert source_hashes(locate_data_dir(DEFAULT_DATA_DIR))==metadata['source_files_sha256']
    assert len(metadata['source_files_sha256'])>144


def test_no_existing_dataset_overwrite(small_export):
    directory,_=small_export
    with pytest.raises(FileExistsError):
        export_dataset(output_dir=directory,record_ids=['100'])


def test_deterministic_export_same_values_and_metadata(small_export,tmp_path):
    directory,metadata=small_export
    second=tmp_path/'second'
    repeated=export_dataset(output_dir=second,record_ids=['100'])
    a=load_record_shard(directory/'recordings/100.npz')
    b=load_record_shard(second/'recordings/100.npz')
    for key in a:
        np.testing.assert_array_equal(a[key],b[key])
    assert metadata==repeated


@pytest.mark.parametrize('field,bad',[
    ('X',np.zeros((2,2,251),dtype=np.float32)),
    ('X',np.full((2271,2,252),np.nan,dtype=np.float32)),
    ('X',np.zeros((2271,2,252),dtype=np.float64)),
    ('y',np.array([0],dtype=np.int64)),
    ('y',np.full(2271,5,dtype=np.int64)),
    ('patient_groups',np.full(2271,'')),
    ('channel_names',np.array(['MLII'])),
])
def test_invalid_arrays_rejected(small_export,field,bad):
    directory,_=small_export
    arrays=load_record_shard(directory/'recordings/100.npz')
    arrays[field]=bad
    with pytest.raises(ValueError):
        validate_shard(arrays)


def test_missing_required_array(small_export):
    directory,_=small_export
    arrays=load_record_shard(directory/'recordings/100.npz')
    del arrays['original_symbols']
    with pytest.raises(ValueError):
        validate_shard(arrays)


@pytest.mark.parametrize('ids',[[],['999'],['100','100']])
def test_invalid_record_selection(tmp_path,ids):
    with pytest.raises(ValueError):
        export_dataset(output_dir=tmp_path/'invalid',record_ids=ids)


def test_all_48_exported_shards_and_phase9_counts():
    # The phase-10 deliverable must exist; this is an intentional end-to-end check.
    report=verify_export(DEFAULT_OUTPUT_DIR,source_root=DEFAULT_DATA_DIR)
    assert report=={'segment_count':109452,'record_count':48,'patient_group_count':47,
                    'class_counts':PHASE9_COUNTS,'source_files_verified':True}
    with (DEFAULT_OUTPUT_DIR/'dataset_metadata.json').open() as source:
        metadata=json.load(source)
    assert metadata['phase9_counts_match'] is True
    assert metadata['annotation_audit']=={'total_annotations':112647,'excluded_nonheartbeat':3153,'skipped_boundary':42}
    assert sum(metadata['class_counts'].values())==metadata['segment_count']
    first=load_record_shard(DEFAULT_OUTPUT_DIR/'recordings/201.npz')
    second=load_record_shard(DEFAULT_OUTPUT_DIR/'recordings/202.npz')
    assert set(first['patient_groups'])==set(second['patient_groups'])=={'mitdb_group_201_202'}


def test_lead_exceptions_preserved_in_full_export():
    for rid,expected in [('100',['MLII','V5']),('102',['V5','V2']),('104',['V5','V2']),('114',['V5','MLII'])]:
        arrays=load_record_shard(DEFAULT_OUTPUT_DIR/f'recordings/{rid}.npz')
        assert arrays['channel_names'].tolist()==expected
