"""Adapter tests use explicit synthetic WFDB fixtures in tmp_path, never pilot outputs."""
import io
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import wfdb
from src.ptbxl_metadata import validate_metadata_frames,load_ptb_metadata
from src.ptbxl_rhythm import (CLASS_NAMES,CLASS_INDICES,EXPECTED_LEADS,DEFAULT_WAVEFORM_ROOT,
    record_identity,select_pilot,pilot_files,download_pilot,load_rhythm_record,preprocess_rhythm_record,
    export_pilot_record,load_pilot_record,preprocessing_config)


@pytest.fixture
def metadata():
    codes=list(CLASS_NAMES)+['AFLT','SARRH','PACE','SVARR','SVTAC','PSVT','BIGU','TRIGU','NORM']
    definitions=pd.DataFrame({'description':codes,'diagnostic':[0]*12+[1],'form':[0]*13,
                              'rhythm':[1]*12+[0]},index=codes)
    labels=[code for code in CLASS_NAMES for _ in range(2)]
    ids=list(range(90001,90009))
    records=pd.DataFrame({'ecg_id':ids,'patient_id':range(1001,1009),'strat_fold':range(1,9),
        'scp_codes':[str({code:0.0,'NORM':100.0}) for code in labels],
        'filename_lr':[f'records100/90000/{eid}_lr' for eid in ids],
        'filename_hr':[f'records500/90000/{eid}_hr' for eid in ids]})
    return validate_metadata_frames(records,definitions)


def write_fixture(metadata,root,eid=90001,nonfinite=False):
    # Not clinical ECG, not downloaded PTB-XL. Unit-test waveform only.
    identity=record_identity(metadata,eid)
    base=Path(root)/identity['filename_lr']
    base.parent.mkdir(parents=True,exist_ok=True)
    time=np.arange(1000)/100
    values=np.column_stack([0.2*np.sin(2*np.pi*(1+i)*time)+0.05*np.sin(2*np.pi*0.1*time)+0.1 for i in range(12)])
    if nonfinite:
        values[100,0]=np.nan
    wfdb.wrsamp(base.name,fs=100,units=['mV']*12,sig_name=list(EXPECTED_LEADS),
        p_signal=values,fmt=['16']*12,adc_gain=[1000]*12,baseline=[0]*12,write_dir=str(base.parent))
    return base


def test_selection_and_mapping_reproducible(metadata):
    selection=select_pilot(metadata)
    assert [r['ecg_id'] for r in selection]==list(range(90001,90009))
    assert [r['y'] for r in selection]==[0,0,1,1,2,2,3,3]
    assert CLASS_INDICES=={'SR':0,'AFIB':1,'STACH':2,'SBRAD':3}
    assert select_pilot(metadata)==selection
    assert len(pilot_files(selection))==16
    assert all('records100/' in f['url'] and f['relative_path'].endswith(('.dat','.hea')) for f in pilot_files(selection))
    for value in (0,3,True):
        with pytest.raises(ValueError):select_pilot(metadata,value)


def test_exclusive_cohort_and_missing_rhythm_never_SR(metadata):
    for value in ("{'SR':0,'AFIB':0}","{'NORM':100}","{'AFLT':100}"):
        records=metadata.records.copy()
        records.loc[0,'scp_codes']=value
        data=validate_metadata_frames(records,metadata.definitions)
        with pytest.raises(ValueError):record_identity(data,90001)
    with pytest.raises(ValueError):record_identity(metadata,1)


def test_read_preprocess_identity_shape_and_nonmutation(metadata,tmp_path):
    write_fixture(metadata,tmp_path)
    loaded=load_rhythm_record(metadata,90001,tmp_path)
    original=loaded.raw.copy()
    assert loaded.raw.shape==(1000,12)
    assert loaded.sampling_frequency==100
    assert loaded.channel_names==EXPECTED_LEADS
    assert loaded.identity['patient_id']==1001 and loaded.identity['strat_fold']==1
    assert loaded.identity['original_scp_statements']=={'SR':0.,'NORM':100.}
    processed=preprocess_rhythm_record(loaded)
    assert processed.X.shape==(12,1000) and processed.X.dtype==np.float32
    assert processed.X.flags.c_contiguous and np.isfinite(processed.X).all()
    assert processed.y.dtype==np.int64 and processed.y==0
    np.testing.assert_allclose(processed.X.mean(axis=1),0,atol=1e-6)
    np.testing.assert_allclose(processed.X.std(axis=1),1,atol=1e-6)
    np.testing.assert_array_equal(loaded.raw,original)
    np.testing.assert_array_equal(preprocess_rhythm_record(loaded).X,processed.X)
    assert processed.configuration['fitted_normalization']['scope']=='recording'
    assert processed.configuration['fitted_normalization']['source_ids']==['90001']


@pytest.mark.parametrize('hz',[50,60])
def test_notch_at_or_above_nyquist_rejected(metadata,tmp_path,hz):
    write_fixture(metadata,tmp_path)
    loaded=load_rhythm_record(metadata,90001,tmp_path)
    config=preprocessing_config()
    config['filtering']['notch_hz']=hz
    with pytest.raises(ValueError,match='Notch'):
        preprocess_rhythm_record(loaded,config=config)


def test_invalid_lowpass_rejected_by_existing_filter(metadata,tmp_path):
    write_fixture(metadata,tmp_path)
    config=preprocessing_config()
    config['filtering']['lowpass_hz']=50
    with pytest.raises(ValueError,match='Nyquist'):
        preprocess_rhythm_record(load_rhythm_record(metadata,90001,tmp_path),config=config)


def test_export_reload_original_metadata_and_no_overwrite(metadata,tmp_path):
    write_fixture(metadata,tmp_path)
    processed=preprocess_rhythm_record(load_rhythm_record(metadata,90001,tmp_path))
    path=tmp_path/'pilot.npz'
    export_pilot_record(processed,path)
    data=load_pilot_record(path)
    np.testing.assert_array_equal(data['X'],processed.X)
    assert data['y'].shape==() and data['y'].dtype==np.int64
    assert int(data['ecg_id'])==90001 and int(data['patient_id'])==1001 and int(data['strat_fold'])==1
    assert json.loads(str(data['original_scp_codes']))=={'SR':0,'NORM':100}
    assert data['channel_names'].tolist()==list(EXPECTED_LEADS)
    with pytest.raises(FileExistsError):export_pilot_record(processed,path)
    data['patient_id']=np.array(999,dtype=np.int64)
    wrong=tmp_path/'wrong.npz'
    np.savez(wrong,**data)
    with pytest.raises(ValueError,match='metadata'):
        load_pilot_record(wrong)


def test_missing_waveforms_have_explicit_paths(metadata,tmp_path):
    with pytest.raises(FileNotFoundError,match='90001_lr.hea'):
        load_rhythm_record(metadata,90001,tmp_path)


@pytest.mark.parametrize('change',['identity','fs','length','leads','units'])
def test_invalid_header_refused(metadata,tmp_path,change):
    base=write_fixture(metadata,tmp_path)
    path=base.with_suffix('.hea')
    lines=path.read_text().splitlines()
    first=lines[0].split()
    if change=='identity':first[0]='90002_lr'
    if change=='fs':first[2]='500'
    if change=='length':first[3]='999'
    lines[0]=' '.join(first)
    if change=='leads':lines[1]=lines[1].rsplit(' ',1)[0]+' II'
    if change=='units':lines[1]=lines[1].replace('/mV','/uV')
    path.write_text('\n'.join(lines)+'\n')
    with pytest.raises(ValueError):load_rhythm_record(metadata,90001,tmp_path)


@pytest.mark.parametrize('corruption',['truncated','checksum'])
def test_corrupted_data_refused(metadata,tmp_path,corruption):
    base=write_fixture(metadata,tmp_path)
    path=base.with_suffix('.dat')
    payload=bytearray(path.read_bytes())
    if corruption=='truncated':payload=payload[:-1]
    else:payload[0]^=1
    path.write_bytes(payload)
    with pytest.raises(ValueError):load_rhythm_record(metadata,90001,tmp_path)


def test_nonfinite_physical_ecg_refused(metadata,tmp_path):
    write_fixture(metadata,tmp_path,nonfinite=True)
    with pytest.raises(ValueError,match='nonfinite'):
        load_rhythm_record(metadata,90001,tmp_path)


def test_downloader_requests_selected_files_only_and_never_overwrites(metadata,tmp_path,monkeypatch):
    import src.ptbxl_rhythm as module
    selection=select_pilot(metadata,1)
    requests=[]
    def fetch(url,timeout):
        requests.append(url)
        return io.BytesIO(b'test fixture bytes only')
    monkeypatch.setattr(module,'urlopen',fetch)
    results=download_pilot(selection,tmp_path)
    assert len(requests)==8 and all(r['status']=='downloaded' for r in results)
    assert all(url.startswith(module.OFFICIAL_BASE_URL+'records100/') for url in requests)
    again=download_pilot(selection,tmp_path)
    assert len(requests)==8 and all(r['status']=='already_present' for r in again)


def test_downloader_network_failure_leaves_no_fabricated_file(metadata,tmp_path,monkeypatch):
    import src.ptbxl_rhythm as module
    def fail(*args,**kwargs):raise OSError('network unavailable in test')
    monkeypatch.setattr(module,'urlopen',fail)
    results=download_pilot(select_pilot(metadata,1),tmp_path)
    assert all(r['status']=='download_failed' for r in results)
    assert not list(tmp_path.rglob('*.dat')) and not list(tmp_path.rglob('*.hea'))


def test_stage_plot_alignment_using_fixture_only(metadata,tmp_path):
    from examples.prepare_ptbxl_pilot import plot_pilot_record
    import matplotlib.pyplot as plt
    write_fixture(metadata,tmp_path)
    processed=preprocess_rhythm_record(load_rhythm_record(metadata,90001,tmp_path))
    fig=plot_pilot_record(processed)
    try:
        assert len(fig.axes)==12
        for ax in fig.axes:
            np.testing.assert_array_equal(ax.lines[0].get_xdata(),np.arange(1000)/100)
            assert ax.get_xlim()==(0,10)
    finally:
        plt.close(fig)


def test_actual_metadata_selection_is_frozen():
    data=load_ptb_metadata()
    selected=select_pilot(data)
    assert [r['ecg_id'] for r in selected]==[1,3,152,282,103,108,2,12]
    assert [r['patient_id'] for r in selected]==[15709,20372,9796,437,13722,11810,13243,11031]
    assert [r['strat_fold'] for r in selected]==[3,5,9,5,4,8,2,8]


def test_real_selected_waveforms_if_present():
    # Explicitly skipped when downloads are unavailable, never replaced with fixtures.
    ids=[1,3,152,282,103,108,2,12]
    expected=[DEFAULT_WAVEFORM_ROOT/'records100/00000'/f'{eid:05d}_lr{suffix}' for eid in ids for suffix in ('.hea','.dat')]
    if any(not path.is_file() for path in expected):
        pytest.skip('Actual selected PTB-XL waveform files are missing; no synthetic fallback')
    data=load_ptb_metadata()
    for eid in ids:
        result=preprocess_rhythm_record(load_rhythm_record(data,eid))
        assert result.X.shape==(12,1000) and np.isfinite(result.X).all()
