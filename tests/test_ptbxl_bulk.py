"""Controlled fixtures test recovery mechanics, not clinical waveform validity."""
import io
import json
from pathlib import Path
import numpy as np
import pytest
from test_ptbxl_rhythm import metadata, write_fixture
from src.ptbxl_metadata import validate_metadata_frames,load_ptb_metadata
from src.ptbxl_rhythm import load_rhythm_record,preprocess_rhythm_record,load_pilot_record
from src.ptbxl_bulk import (reconstruct_cohort,required_files,dry_run,fetch_file,
    ensure_waveform,export_or_validate,run_export,distribution_tables,assert_patient_folds,
    exclusive_run_lock,install_exclusive)


class Response(io.BytesIO):
    def __init__(self,content,status=200,headers=None):
        super().__init__(content)
        self.status=status
        self.headers={'Content-Length':str(len(content)),**(headers or {})}


def item(size=24000):
    return {'relative_path':'records100/90000/90001_lr.dat',
            'url':'https://physionet-open.s3.amazonaws.com/ptb-xl/1.0.3/records100/90000/90001_lr.dat',
            'expected_bytes':size}


def cohort(metadata):
    return reconstruct_cohort(metadata,verify_phase12=False)


def test_independent_reconstruction_and_order(metadata):
    rows=cohort(metadata)
    assert [r['ecg_id'] for r in rows]==list(range(90001,90009))
    assert [r['y'] for r in rows]==[0,0,1,1,2,2,3,3]
    assert len(list(required_files(rows)))==16
    assert all('/records100/' in x['url'] for x in required_files(rows))
    with pytest.raises(ValueError,match='discrepancy'): reconstruct_cohort(metadata)


@pytest.mark.parametrize('statements',["{'SR':0,'PACE':0}","{'NORM':100}","{'AFLT':100}","{'SR':0,'AFIB':100}"])
def test_all_twelve_codes_and_unknown_not_normal(metadata,statements):
    frame=metadata.records.copy()
    frame.loc[0,'scp_codes']=statements
    data=validate_metadata_frames(frame,metadata.definitions)
    assert 90001 not in [r['ecg_id'] for r in cohort(data)]
    assert len(cohort(data))==7


def test_fold_conflict_blocks_before_download(metadata):
    frame=metadata.records.copy()
    frame.loc[1,'patient_id']=frame.loc[0,'patient_id']
    data=validate_metadata_frames(frame,metadata.definitions)
    with pytest.raises(ValueError,match='multiple folds'): assert_patient_folds(data)


def test_filename_safety(metadata):
    row=cohort(metadata)[0]
    for path in ('../bad','records500/90000/90001_lr','records100/90000/90002_lr'):
        with pytest.raises(ValueError): list(required_files([{**row,'filename_lr':path}]))
    with pytest.raises(ValueError): list(required_files([row,row]))


def test_dry_run_never_network_or_write(metadata,tmp_path,monkeypatch):
    import urllib.request
    monkeypatch.setattr(urllib.request,'urlopen',lambda *a,**k:pytest.fail('network'))
    result=dry_run(metadata,waveform_root=tmp_path,disk_root=tmp_path,verify_phase12=False)
    assert list(tmp_path.iterdir())==[]
    assert result['recordings']==8 and result['required_files']==16
    assert result['dat_bytes_full_cohort']==8*24000
    assert result['total_download_bytes'] is None
    assert result['uncompressed_X_bytes']==8*48000
    assert result['disk_free_bytes']>0 and result['bulk_started'] is False


def test_dry_run_counts_real_validated_local_fixture(metadata,tmp_path):
    write_fixture(metadata,tmp_path)
    result=dry_run(metadata,waveform_root=tmp_path,disk_root=tmp_path,verify_phase12=False)
    assert result['files_in_validated_pairs']==2
    assert result['dat_bytes_not_in_validated_pairs']==7*24000


def test_retry_and_exponential_backoff(tmp_path):
    calls=[]; sleeps=[]
    def opener(req,**kwargs):
        calls.append(req.full_url)
        if len(calls)<3: raise OSError('controlled offline')
        return Response(b'x'*24000)
    path=fetch_file(item(),tmp_path,opener=opener,sleep=sleeps.append)
    assert path.read_bytes()==b'x'*24000 and sleeps==[1,2]
    assert len(calls)==3 and not path.with_suffix('.dat.part').exists()


def test_network_failure_retains_partial_then_range_resume(tmp_path):
    calls=[]
    def short(req,**kwargs):
        return Response(b'a'*100,headers={'Content-Length':'24000'})
    with pytest.raises(ValueError,match='Truncated'):
        fetch_file(item(),tmp_path,retries=1,opener=short)
    path=tmp_path/item()['relative_path']
    assert not path.exists() and path.with_suffix('.dat.part').stat().st_size==100
    def resume(req,**kwargs):
        calls.append(req.get_header('Range'))
        return Response(b'b'*23900,206,{'Content-Range':'bytes 100-23999/24000'})
    fetch_file(item(),tmp_path,opener=resume)
    assert calls==['bytes=100-']
    assert path.read_bytes()==b'a'*100+b'b'*23900


def test_ignored_range_restarts_staging(tmp_path):
    path=tmp_path/item()['relative_path']
    path.parent.mkdir(parents=True)
    path.with_suffix('.dat.part').write_bytes(b'old')
    fetch_file(item(),tmp_path,opener=lambda *a,**k:Response(b'b'*24000))
    assert path.read_bytes()==b'b'*24000


def test_invalid_content_range_preserves_partial(tmp_path):
    path=tmp_path/item()['relative_path']; path.parent.mkdir(parents=True)
    part=path.with_suffix('.dat.part'); part.write_bytes(b'old')
    with pytest.raises(ValueError,match='Content-Range'):
        fetch_file(item(),tmp_path,retries=1,opener=lambda *a,**k:Response(b'x',206,{'Content-Range':'bytes 9-9/10'}))
    assert part.read_bytes()==b'old' and not path.exists()


def test_oversized_payload_not_installed(tmp_path):
    with pytest.raises(ValueError,match='large'):
        fetch_file(item(),tmp_path,retries=1,opener=lambda *a,**k:Response(b'x'*24001))
    assert not (tmp_path/item()['relative_path']).exists()


def test_invalid_staging_size_rejected(tmp_path):
    path=tmp_path/item()['relative_path']; path.parent.mkdir(parents=True); path.write_bytes(b'bad')
    with pytest.raises(ValueError,match='staging size'): fetch_file(item(),tmp_path)


def test_download_validate_install_and_skip(metadata,tmp_path):
    source=tmp_path/'source'; write_fixture(metadata,source)
    row=cohort(metadata)[0]; root=tmp_path/'raw'; stage=root/'.staging'
    calls=[]
    def opener(req,**kwargs):
        calls.append(req.full_url)
        return Response((source/req.full_url.split('/1.0.3/')[1]).read_bytes())
    loaded,status=ensure_waveform(metadata,row,root,stage,opener=opener)
    assert status=='downloaded_or_resumed' and len(calls)==2
    assert loaded.identity['patient_id']==1001
    original=[p.stat().st_mtime_ns for p in (root/'records100/90000').iterdir()]
    _,status=ensure_waveform(metadata,row,root,stage,opener=lambda *a,**k:pytest.fail('redownload'))
    assert status=='validated_existing'
    assert original==[p.stat().st_mtime_ns for p in (root/'records100/90000').iterdir()]


def test_corrupt_download_never_installed(metadata,tmp_path):
    source=tmp_path/'source'; base=write_fixture(metadata,source)
    payload=bytearray(base.with_suffix('.dat').read_bytes()); payload[30]^=1
    def opener(req,**kwargs):
        return Response(bytes(payload) if req.full_url.endswith('.dat') else base.with_suffix('.hea').read_bytes())
    root=tmp_path/'raw'
    with pytest.raises(ValueError,match='checksum'):
        ensure_waveform(metadata,cohort(metadata)[0],root,root/'.stage',opener=opener)
    assert not (root/cohort(metadata)[0]['filename_lr']).with_suffix('.dat').exists()


def test_existing_corrupt_pair_never_overwritten(metadata,tmp_path):
    base=write_fixture(metadata,tmp_path)
    base.with_suffix('.dat').write_bytes(b'corrupt')
    with pytest.raises(ValueError):
        ensure_waveform(metadata,cohort(metadata)[0],tmp_path,tmp_path/'.stage')
    assert base.with_suffix('.dat').read_bytes()==b'corrupt'


def test_export_roundtrip_resume_and_changed_metadata(metadata,tmp_path):
    write_fixture(metadata,tmp_path/'raw')
    processed=preprocess_rhythm_record(load_rhythm_record(metadata,90001,tmp_path/'raw'))
    path=tmp_path/'out/90001.npz'; pending=tmp_path/'pending'
    first=export_or_validate(processed,path,pending)
    before=path.read_bytes()
    second=export_or_validate(processed,path,pending)
    assert first['sha256']==second['sha256'] and path.read_bytes()==before
    assert second['status']=='validated_existing_export'
    data=load_pilot_record(path)
    assert data['X'].dtype==np.float32 and data['X'].shape==(12,1000)
    assert data['y'].dtype==np.int64 and data['patient_id']==1001
    assert data['strat_fold']==1
    assert json.loads(str(data['original_scp_codes']))=={'NORM':100.,'SR':0.}
    processed.loaded.identity['patient_id']=999
    with pytest.raises(ValueError,match='differs'): export_or_validate(processed,path,pending)
    assert path.read_bytes()==before


def test_approval_guard_before_side_effects(metadata,tmp_path):
    with pytest.raises(PermissionError): run_export(metadata,output_dir=tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_full_fixture_export_resume_partial_failure_and_provenance(metadata,tmp_path):
    raw=tmp_path/'raw'; output=tmp_path/'export'
    # Exactly one available fixture; no network. Remaining seven remain failures.
    write_fixture(metadata,raw)
    options={'opener':lambda *a,**k:(_ for _ in ()).throw(OSError('offline fixture')),'retries':1}
    directory,result=run_export(metadata,approved=True,waveform_root=raw,output_dir=output,
        verify_phase12=False,download_options=options)
    assert result['successful_records']==1 and result['failed_records']==7
    assert result['logical_X_shape']==[1,12,1000] and result['status']=='incomplete'
    assert result['records'][0]['ecg_id']==90001
    assert [r['ecg_id'] for r in result['records']]==list(range(90001,90009))
    assert json.loads((directory/'PTBXL_DOWNLOAD_MANIFEST.json').read_text())==result
    path=output/result['records'][0]['file']; original=path.read_bytes()
    directory2,result2=run_export(metadata,approved=True,waveform_root=raw,output_dir=output,
        verify_phase12=False,download_options=options)
    assert directory!=directory2 and result2['records'][0]['status']=='validated_existing_export'
    assert path.read_bytes()==original
    identity=output/'dataset_identity.json'
    contents=json.loads(identity.read_text()); contents['dataset_version']='other'; identity.write_text(json.dumps(contents))
    with pytest.raises(ValueError,match='identity changed'):
        run_export(metadata,approved=True,waveform_root=raw,output_dir=output,verify_phase12=False)


def test_distribution_separates_selected_and_actual(metadata):
    rows=cohort(metadata); classes,folds=distribution_tables(rows,rows[:1])
    assert sum(r['selected_records'] for r in classes)==8
    assert sum(r['exported_records'] for r in classes)==1
    assert sum(r['exported_records'] for r in folds)==1
    assert len(folds)==40


def test_lock_and_exclusive_install(tmp_path):
    with exclusive_run_lock(tmp_path/'lock'):
        with pytest.raises(FileExistsError):
            with exclusive_run_lock(tmp_path/'lock'): pass
    assert not (tmp_path/'lock').exists()
    source=tmp_path/'source'; source.write_text('data')
    install_exclusive(source,tmp_path/'dest')
    with pytest.raises(FileExistsError): install_exclusive(source,tmp_path/'dest')


def test_real_metadata_cohort_and_local_pilot():
    data=load_ptb_metadata(); rows=reconstruct_cohort(data)
    assert len(rows)==19605 and len({r['patient_id'] for r in rows})==17223
    assert sum(r['label']=='AFIB' for r in rows)==1484
    assert len(assert_patient_folds(data))==18869


def test_missing_header_recovers_without_overwriting_existing_dat(metadata,tmp_path):
    source=tmp_path/'source'; base=write_fixture(metadata,source)
    raw=tmp_path/'raw'; destination=raw/cohort(metadata)[0]['filename_lr']
    destination.parent.mkdir(parents=True)
    destination.with_suffix('.dat').write_bytes(base.with_suffix('.dat').read_bytes())
    before=destination.with_suffix('.dat').stat().st_mtime_ns
    calls=[]
    def opener(req,**kwargs):
        calls.append(req.full_url)
        assert req.full_url.endswith('.hea')
        return Response(base.with_suffix('.hea').read_bytes())
    loaded,status=ensure_waveform(metadata,cohort(metadata)[0],raw,raw/'.stage',opener=opener)
    assert len(calls)==1 and loaded.raw.shape==(1000,12)
    assert destination.with_suffix('.dat').stat().st_mtime_ns==before


def test_complete_controlled_export_reports_actual_counts(metadata,tmp_path):
    raw=tmp_path/'raw'
    for row in cohort(metadata): write_fixture(metadata,raw,row['ecg_id'])
    run,result=run_export(metadata,approved=True,waveform_root=raw,output_dir=tmp_path/'out',verify_phase12=False)
    assert result['status']=='complete' and result['successful_records']==8
    assert result['logical_X_shape']==[8,12,1000]
    assert result['failed_records']==0
    assert all(r['exported_records']==2 for r in result['class_distribution'])
    assert 'Validated exported records: 8' in (run/'PTBXL_FULL_EXPORT_REPORT.md').read_text()


def test_stop_on_error_preserves_progress_and_reports_unattempted(metadata,tmp_path):
    raw=tmp_path/'raw'; output=tmp_path/'out'
    write_fixture(metadata,raw)
    calls=[]
    def offline(*args,**kwargs):
        calls.append(1)
        raise OSError('controlled network failure')
    directory,result=run_export(metadata,approved=True,waveform_root=raw,output_dir=output,
        verify_phase12=False,download_options={'opener':offline,'retries':1},stop_on_error=True)
    assert len(calls)==1
    assert result['successful_records']==1 and result['failed_records']==1
    assert result['not_attempted_records']==6
    assert result['stop_reason']=='OSError: controlled network failure'
    assert len(result['records'])==8
    assert result['records'][0]['status']=='exported'
    assert all(row['status']=='not_attempted' for row in result['records'][2:])
    assert (output/result['records'][0]['file']).exists()
    assert not (output/'.run.lock').exists()
    assert not (raw/'.phase14_staging/.download.lock').exists()
    assert 'Not attempted: 6' in (directory/'PTBXL_FULL_EXPORT_REPORT.md').read_text()
