"""Validate generated figures against real local data; no synthetic waveforms."""
from pathlib import Path
import json
import struct
import numpy as np
import pytest
import wfdb
from src.data_loader import DEFAULT_DATA_DIR
from src.prepare_dataset import load_record_shard,sha256_file
from src.label_mapping import DEFAULT_MAPPING
from src.ptbxl_metadata import load_ptb_metadata
from src.ptbxl_rhythm import record_identity,load_pilot_record

ROOT=Path('results/preprocessing/final_visual_validation')


@pytest.fixture(scope='module')
def visual():
    path=ROOT/'visual_validation_manifest.json'
    if not path.exists():
        pytest.skip('Generate the real-data visual validation artifacts first')
    return json.loads(path.read_text())


def test_complete_figure_set(visual):
    assert visual['status']=='passed' and visual['figure_count']==8
    assert sum(r['kind']=='mit_stages' for r in visual['figures'])==2
    assert sum(r['kind']=='ptb_stages' for r in visual['figures'])==4
    assert sum(r['kind']=='mit_beats' for r in visual['figures'])==1
    assert sum(r['kind']=='ptb_overview' for r in visual['figures'])==1


@pytest.mark.parametrize('index',range(8))
def test_publication_formats_and_nonempty_captions(visual,index):
    item=visual['figures'][index]
    assert len(item['caption'])>80
    png,pdf=map(Path,item['paths'])
    header=png.read_bytes()[:24]
    assert header[:8]==b'\x89PNG\r\n\x1a\n'
    width,height=struct.unpack('>II',header[16:24])
    assert width>=5500 and height>=2200
    assert pdf.read_bytes().startswith(b'%PDF') and pdf.stat().st_size>10000


def test_mit_examples_match_expert_annotations_and_export(visual):
    entry=next(r for r in visual['figures'] if r['kind']=='mit_beats')
    assert set(entry['examples'])==set('NSVFQ')
    for code,row in entry['examples'].items():
        rid=row['record_id'];archive=load_record_shard(f'data/processed/ecg_dataset_v1/recordings/{rid}.npz')
        annotation=wfdb.rdann(str(DEFAULT_DATA_DIR/rid),'atr')
        ai=row['annotation_index'];index=row['export_row']
        assert annotation.sample[ai]==row['position']==archive['annotation_positions'][index]
        assert annotation.symbol[ai]==row['original_symbol']==archive['original_symbols'][index]
        assert DEFAULT_MAPPING[row['original_symbol']]==code
        assert int(archive['y'][index])==row['class_index']
        assert row['channels']==archive['channel_names'].tolist()
        assert archive['X'][index].shape==(2,252)
        assert row['before_samples']==90 and row['sampling_frequency']==360
        assert sha256_file(Path(f'data/processed/ecg_dataset_v1/recordings/{rid}.npz'))==row['export_sha256']


def test_ptb_four_exclusive_labels_and_metadata(visual):
    data=load_ptb_metadata()
    rows=[r for r in visual['figures'] if r['kind']=='ptb_stages']
    assert {r['rhythm_code'] for r in rows}=={'SR','AFIB','STACH','SBRAD'}
    for row in rows:
        identity=record_identity(data,row['ecg_id'])
        assert set(identity['original_scp_statements'])&set(data.rhythm_codes)=={row['rhythm_code']}
        assert row['original_scp_statements']==identity['original_scp_statements']
        assert row['patient_id']==identity['patient_id'] and row['strat_fold']==identity['strat_fold']
        path=Path(f"data/processed/ptbxl_rhythm_v1/recordings/{row['ecg_id']:05d}_rhythm.npz")
        archive=load_pilot_record(path)
        assert int(archive['y'])==row['class_index']==identity['y']
        assert row['displayed_leads']==['II','V1','V5']
        assert archive['X'].shape==(12,1000) and float(archive['sampling_frequency'])==100
        assert row['export_sha256']==sha256_file(path)


def test_mit_stage_time_coordinates_and_units(visual):
    for row in (r for r in visual['figures'] if r['kind']=='mit_stages'):
        assert row['start_sample']==3600 and row['stop_sample_exclusive']==7200
        annotation=wfdb.rdann(str(DEFAULT_DATA_DIR/row['record_id']),'atr')
        for position,symbol in zip(row['annotation_positions'],row['annotation_symbols']):
            assert 3600<=position<7200
            assert any(p==position and s==symbol for p,s in zip(annotation.sample,annotation.symbol))
        header=wfdb.rdheader(str(DEFAULT_DATA_DIR/row['record_id']))
        assert header.units==['mV','mV']
        assert row['channel_names']==header.sig_name


def test_verified_real_source_checksums_and_unmodified_code(visual):
    for path,digest in visual['sources_sha256'].items():assert sha256_file(path)==digest
    for path,digest in visual['code_sha256'].items():assert sha256_file(path)==digest
