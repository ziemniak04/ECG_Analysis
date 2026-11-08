"""Phase-14 bounded-memory PTB-XL export. Dry-run is strictly network-free.

Immutable waveform/export destinations; resumable staging files belong to this
module. A run lock excludes concurrent writers. No patient splits or balancing.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
import csv
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import shutil
import time
from urllib.request import Request, urlopen
from uuid import uuid4

import numpy as np
from .ptbxl_metadata import load_ptb_metadata, DEFAULT_METADATA_DIR, propose_exclusive_cohort
from .ptbxl_rhythm import (CLASS_NAMES, DEFAULT_WAVEFORM_ROOT, record_identity,
    load_rhythm_record, preprocess_rhythm_record, preprocessing_config,
    export_pilot_record, load_pilot_record)
from .prepare_dataset import sha256_file

S3_BASE = 'https://physionet-open.s3.amazonaws.com/ptb-xl/1.0.3/'
VERSION = 'ptbxl-rhythm-v1'
EXPECTED_COUNTS = {'SR':16687, 'AFIB':1484, 'STACH':805, 'SBRAD':629}
LOG = logging.getLogger(__name__)


def write_new_json(path, value):
    """Never replace a pre-existing report or export."""
    with Path(path).open('x', encoding='utf-8') as target:
        json.dump(value, target, indent=2, sort_keys=True, allow_nan=False)


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def assert_patient_folds(data):
    """Audit all original patients, without constructing model splits."""
    mapping = {}
    for patient, fold in zip(data.patient_ids, data.folds):
        patient, fold = int(patient), int(fold)
        if patient in mapping and mapping[patient] != fold:
            raise ValueError(f'Patient {patient} occurs in multiple folds')
        mapping[patient] = fold
    return mapping


def reconstruct_cohort(data, *, verify_phase12=True):
    """Independently select exactly one of ALL 12 rhythm keys; cross-check audit."""
    assert_patient_folds(data)
    if len(data.rhythm_codes) != 12:
        raise ValueError('Expected all 12 rhythm definitions')
    rhythms = set(data.rhythm_codes)
    selected = []
    for eid, statements in zip(data.ecg_ids, data.statements):
        present = set(statements) & rhythms
        if len(present) == 1 and next(iter(present)) in CLASS_NAMES:
            selected.append(record_identity(data, int(eid)))
    selected.sort(key=lambda row: row['ecg_id'])
    original = propose_exclusive_cohort(data)['record_audit']
    if [r['ecg_id'] for r in selected] != sorted(r['ecg_id'] for r in original if r['proposed_label'] in CLASS_NAMES):
        raise ValueError('Independent cohort differs from Phase-12 implementation')
    counts = dict(Counter(row['label'] for row in selected))
    if verify_phase12 and (len(selected) != 19605 or len({r['patient_id'] for r in selected}) != 17223 or counts != EXPECTED_COUNTS):
        raise ValueError(f'Phase-12 discrepancy: {len(selected)} records, counts={counts}')
    return selected


def required_files(cohort):
    """Paths derive exclusively from validated metadata, never sequential ID guesses."""
    if len({r['ecg_id'] for r in cohort}) != len(cohort):
        raise ValueError('Duplicate cohort identity')
    for row in cohort:
        base = row['filename_lr']
        match = re.fullmatch(r'records100/\d{5}/(\d{5})_lr', base)
        if not match or int(match[1]) != row['ecg_id']:
            raise ValueError('Unsafe or mismatched metadata path')
        for suffix in ('.hea', '.dat'):
            yield {'ecg_id': row['ecg_id'], 'relative_path': base + suffix,
                   'url': S3_BASE + base + suffix,
                   'expected_bytes': 24000 if suffix == '.dat' else None}


def local_inventory(data, cohort, root):
    """Only fully WFDB-validated pairs are counted as reusable."""
    valid, failed, missing = [], [], []
    root = Path(root)
    for row in cohort:
        paths = [root/(row['filename_lr']+suffix) for suffix in ('.hea','.dat')]
        if not all(path.is_file() for path in paths):
            missing.append(row['ecg_id'])
        else:
            try:
                load_rhythm_record(data, row['ecg_id'], root)
                valid.append(row['ecg_id'])
            except (ValueError, OSError) as exc:
                failed.append({'ecg_id':row['ecg_id'], 'reason':str(exc)})
    return {'validated_record_ids':valid, 'missing_or_incomplete_record_ids':missing,
            'invalid_existing_records':failed}


def dry_run(data, *, waveform_root=DEFAULT_WAVEFORM_ROOT, disk_root=Path('.'), verify_phase12=True):
    """Inspect disk and local waveforms, without URL requests or output writes.

    Exact dat bytes follow the validated format-16 contract. Header sizes and
    NPZ compression/JSON overhead remain unknown; no invented total estimate.
    """
    cohort = reconstruct_cohort(data, verify_phase12=verify_phase12)
    files = list(required_files(cohort))
    inventory = local_inventory(data, cohort, waveform_root)
    valid = set(inventory['validated_record_ids'])
    dat_bytes = len(cohort)*24000
    remaining_dat = sum(item['expected_bytes'] or 0 for item in files
                        if item['ecg_id'] not in valid)
    x_bytes = len(cohort)*12*1000*4
    free = shutil.disk_usage(disk_root).free
    return {'stage':'checkpoint_a_dry_run_no_network', 'bulk_started':False,
        'recordings':len(cohort), 'patients':len({r['patient_id'] for r in cohort}),
        'class_counts':dict(Counter(r['label'] for r in cohort)),
        'required_files':len(files), 'files_in_validated_pairs':2*len(valid),
        'files_not_in_validated_pairs':len(files)-2*len(valid),
        'dat_bytes_full_cohort':dat_bytes, 'dat_bytes_not_in_validated_pairs':remaining_dat,
        'header_download_bytes':None, 'total_download_bytes':None,
        'uncompressed_X_bytes':x_bytes, 'uncompressed_y_bytes':len(cohort)*8,
        'known_disk_budget_component_bytes':remaining_dat+x_bytes+len(cohort)*8,
        'disk_budget_note':'Known components only; headers, JSON, NPZ overhead, filesystem blocks and staging add space. Compression unknown.',
        'disk_free_bytes':free, 'known_components_fit':free > remaining_dat+x_bytes+len(cohort)*8,
        'patient_fold_consistent':True, 'local_inventory':inventory,
        'files':files, 'source':S3_BASE, 'metadata_provenance':data.provenance,
        'preprocessing_configuration':preprocessing_config()}


@contextmanager
def exclusive_run_lock(path):
    """Cooperative local lock; stale locks require explicit operator inspection."""
    path = Path(path)
    with path.open('x') as target:
        target.write(str(os.getpid()))
    try:
        yield
    finally:
        path.unlink()


def fetch_file(item, staging_root, *, retries=3, timeout=30, opener=urlopen, sleep=time.sleep):
    """Resume .part via HTTP Range. Untrusted bytes never enter waveform root.

    A complete .dat must match 24000 bytes, a header <=64 KiB. Full record
    checksum/header validation happens before installation. Range-ignoring
    servers restart the staging file; 206 requires a correct Content-Range.
    No AWS credentials or CLI. Timeout applies per socket operation.
    """
    if retries < 1 or timeout <= 0:
        raise ValueError('Positive attempts and timeout required')
    path = Path(staging_root)/item['relative_path']
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix+'.part')
    expected = item['expected_bytes']
    limit = expected if expected is not None else 65536
    if path.exists():
        if expected is not None and path.stat().st_size != expected:
            raise ValueError('Invalid completed staging size; review staging file')
        return path
    for attempt in range(retries):
        offset = part.stat().st_size if part.exists() else 0
        if expected and offset == expected:
            os.replace(part,path)
            return path
        if offset >= limit:
            raise ValueError('Oversized partial file; explicit recovery required')
        headers = {'Range':f'bytes={offset}-'} if offset else {}
        try:
            with opener(Request(item['url'], headers=headers), timeout=timeout) as response:
                status = response.status
                if status not in (200,206):
                    raise ValueError(f'Unexpected HTTP status {status}')
                append = offset > 0 and status == 206
                if status == 206:
                    content_range = response.headers.get('Content-Range','')
                    match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', content_range)
                    if not match or int(match[1]) != offset or int(match[2]) != int(match[3])-1:
                        raise ValueError('Invalid Content-Range')
                    if expected is not None and int(match[3]) != expected:
                        raise ValueError('Remote size disagrees with waveform contract')
                length = response.headers.get('Content-Length')
                received = 0
                with part.open('ab' if append else 'wb') as target:
                    while True:
                        chunk = response.read(65536)
                        if not chunk:
                            break
                        if target.tell()+len(chunk) > limit:
                            raise ValueError('Unexpectedly large payload')
                        target.write(chunk)
                        received += len(chunk)
                    target.flush()
                    os.fsync(target.fileno())
                if length is not None and received != int(length):
                    raise ValueError('Truncated HTTP payload')
                size = part.stat().st_size
                if not size or (expected is not None and size != expected):
                    raise ValueError('Incomplete or unexpected file size')
                os.replace(part,path)
                return path
        except Exception as exc:
            LOG.warning('Download %s attempt %s/%s: %s',item['relative_path'],attempt+1,retries,exc)
            if attempt+1 == retries:
                raise
            sleep(min(2**attempt,30))
    raise RuntimeError('Unreachable download state')


def install_exclusive(source, destination):
    """Atomic no-clobber installation on the same filesystem."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.link(source, destination)  # fails if destination exists; no replacement


def ensure_waveform(data, row, waveform_root, staging_root, **download_options):
    """Validate existing pair or stage/download/validate a missing pair.

    Existing corrupt files are never replaced. They must be quarantined by an
    operator after review. Staging survives network failures for the next run.
    """
    waveform_root, staging_root = Path(waveform_root), Path(staging_root)
    entries = list(required_files([row]))
    if all((waveform_root/e['relative_path']).exists() for e in entries):
        return load_rhythm_record(data,row['ecg_id'],waveform_root), 'validated_existing'
    for item in entries:
        source = waveform_root/item['relative_path']
        staged = staging_root/item['relative_path']
        if source.exists():
            staged.parent.mkdir(parents=True, exist_ok=True)
            if not staged.exists():
                with source.open('rb') as src, staged.open('xb') as dst:
                    shutil.copyfileobj(src,dst)
            elif sha256_file(source) != sha256_file(staged):
                raise ValueError('Existing/staging mismatch; explicit recovery required')
        else:
            fetch_file(item,staging_root,**download_options)
    load_rhythm_record(data,row['ecg_id'],staging_root)
    for item in entries:
        source = waveform_root/item['relative_path']
        staged = staging_root/item['relative_path']
        if not source.exists():
            install_exclusive(staged,source)
    loaded = load_rhythm_record(data,row['ecg_id'],waveform_root)
    return loaded, 'downloaded_or_resumed'


def export_or_validate(processed, path, pending_root):
    """Recompute and compare every field on resume; never rewrite valid exports."""
    path, pending_root = Path(path), Path(pending_root)
    pending_root.mkdir(parents=True, exist_ok=True)
    temporary = pending_root/(uuid4().hex+'.npz')
    try:
        export_pilot_record(processed,temporary)
        if path.exists():
            old, new = load_pilot_record(path), load_pilot_record(temporary)
            if old.keys() != new.keys() or any(not np.array_equal(old[k],new[k]) for k in old):
                raise ValueError('Existing export differs from recomputed source/configuration')
            status = 'validated_existing_export'
        else:
            install_exclusive(temporary,path)
            status = 'exported'
        return {'status':status, 'sha256':sha256_file(path)}
    finally:
        if temporary.exists():
            temporary.unlink()


def distribution_tables(selected, successful):
    """Separate selected-cohort counts from actual validated export counts."""
    classes, folds = [], []
    for code in CLASS_NAMES:
        a = [r for r in selected if r['label']==code]
        b = [r for r in successful if r['label']==code]
        classes.append({'class':code,'class_index':CLASS_NAMES.index(code),
            'selected_records':len(a),'selected_patients':len({r['patient_id'] for r in a}),
            'exported_records':len(b),'exported_patients':len({r['patient_id'] for r in b}),
            'not_exported':len(a)-len(b)})
        for fold in range(1,11):
            aa=[r for r in a if r['strat_fold']==fold]
            bb=[r for r in b if r['strat_fold']==fold]
            folds.append({'class':code,'strat_fold':fold,'selected_records':len(aa),
                'selected_patients':len({r['patient_id'] for r in aa}),
                'exported_records':len(bb),'exported_patients':len({r['patient_id'] for r in bb})})
    return classes,folds


def write_csv_new(path, rows):
    with Path(path).open('x',newline='') as target:
        writer=csv.DictWriter(target,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)



def write_run_report(path, manifest):
    """Record observed success/failure counts, never relabel a partial export complete."""
    rows = '\n'.join(f"| {r['class']} | {r['selected_records']} | {r['exported_records']} | {r['exported_patients']} |" for r in manifest['class_distribution'])
    report = f"""# PTB-XL export — observed run report

Status: {manifest['status']}. Selected records: {manifest['selected_records']}.
Validated exported records: {manifest['successful_records']}; patients: {manifest['successful_patients']}.
Failed records: {manifest['failed_records']}. Not attempted: {manifest.get('not_attempted_records',0)}.
Stop reason: {manifest.get('stop_reason')}. Actual logical X shape: {manifest['logical_X_shape']}.

| Rhythm | Selected | Exported | Exported patients |
|---|---:|---:|---:|
{rows}

See PTBXL_DOWNLOAD_MANIFEST.json for every recording, source/output hashes and
failure reason. CSVs distinguish selected and exported counts per class/fold.
The manifest orders records by ecg_id. Failed records remain selected and are
never replaced. Resume by rerunning the approved command with the same paths.
Each run retains separate logs/reports; exports are immutable and revalidated.
Metadata provenance and exact processing/code configuration are in dataset_identity.json.
No train/validation/test split, balancing or model work was performed.
"""
    with Path(path).open('x') as target:
        target.write(report)


def run_export(data, *, approved=False, waveform_root=DEFAULT_WAVEFORM_ROOT,
               output_dir=Path('data/processed/ptbxl_rhythm_v1'), verify_phase12=True,
               download_options=None, stop_on_error=False):
    """Approval-gated full export, one actual WFDB record at a time.

    A new run directory/log is retained on every rerun. Dataset identity and
    per-record files are immutable. Failed records stay selected, never silently
    replaced; a later run retries and revalidates all records.
    """
    if approved is not True:
        raise PermissionError('Bulk download/export requires explicit user approval')
    cohort=reconstruct_cohort(data,verify_phase12=verify_phase12)
    output_dir=Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    descriptor={'dataset_version':VERSION,'source_version':'1.0.3','source':S3_BASE,
        'metadata_provenance':data.provenance,'cohort_sha256':json_hash(cohort),
        'configuration':preprocessing_config(),
        'processing_code_sha256':{name:sha256_file(Path(__file__).parent/name)
             for name in ('ptbxl_bulk.py','ptbxl_rhythm.py','ptbxl_metadata.py','preprocessing.py')},
        'logical_shape':[len(cohort),12,1000],'ordering':'ascending ecg_id',
        'cohort_selection_rule':'Exactly one of all 12 rhythm codes, in SR/AFIB/STACH/SBRAD; all eligible records',
        'configuration_note':'Unmodified Phase-13 processing config; its pilot selection_rule is provenance only, not the full cohort rule.',
        'note':'Logical shape is selected size; use successful manifest entries for actual N.'}
    with exclusive_run_lock(output_dir/'.run.lock'):
        identity_path=output_dir/'dataset_identity.json'
        if identity_path.exists():
            if json.loads(identity_path.read_text()) != descriptor:
                raise ValueError('Dataset identity changed; use a new versioned directory')
        else:
            if any(output_dir.iterdir()):
                # The lock itself is expected; all other entries need review.
                if any(p.name!='.run.lock' for p in output_dir.iterdir()):
                    raise ValueError('Unrecognized existing output directory')
            write_new_json(identity_path,descriptor)
        run_dir=output_dir/'runs'/uuid4().hex
        run_dir.mkdir(parents=True)
        staging=Path(waveform_root)/'.phase14_staging'
        staging.mkdir(parents=True,exist_ok=True)
        successful, records=[],[]
        stop_reason=None
        with exclusive_run_lock(staging/'.download.lock'):
            with (run_dir/'events.jsonl').open('x',buffering=1) as events:
                for row in cohort:
                    entry={'ecg_id':row['ecg_id'],'patient_id':row['patient_id'],
                           'strat_fold':row['strat_fold'],'label':row['label']}
                    try:
                        loaded,status=ensure_waveform(data,row,waveform_root,staging,**(download_options or {}))
                        entry['waveform_status']=status
                        processed=preprocess_rhythm_record(loaded)
                        relative=f'recordings/{row["ecg_id"]:05d}_rhythm.npz'
                        exported=export_or_validate(processed,output_dir/relative,output_dir/'.pending')
                        entry.update(exported)
                        entry.update({'file':relative,'source_hashes':loaded.source_hashes})
                        successful.append(row)
                    except Exception as exc:
                        entry.update({'status':'failed','reason':f'{type(exc).__name__}: {exc}'})
                        LOG.error('ECG %s failed: %s',row['ecg_id'],exc)
                    records.append(entry)
                    events.write(json.dumps(entry,sort_keys=True)+'\n')
                    events.flush()
                    os.fsync(events.fileno())
                    LOG.info('ECG %s: %s',row['ecg_id'],entry['status'])
                    if stop_on_error and entry['status']=='failed':
                        stop_reason=entry['reason']
                        break
                attempted_ids={entry['ecg_id'] for entry in records}
                for row in cohort:
                    if row['ecg_id'] not in attempted_ids:
                        entry={'ecg_id':row['ecg_id'],'patient_id':row['patient_id'],
                               'strat_fold':row['strat_fold'],'label':row['label'],
                               'status':'not_attempted','reason':'Execution stopped after preceding failure'}
                        records.append(entry)
                        events.write(json.dumps(entry,sort_keys=True)+'\n')
                events.flush()
                os.fsync(events.fileno())
        classes,folds=distribution_tables(cohort,successful)
        manifest={'stage':'checkpoint_b','dataset':descriptor,'selected_records':len(cohort),
            'successful_records':len(successful),'successful_patients':len({r['patient_id'] for r in successful}),
            'downloaded_or_resumed_records':sum(r.get('waveform_status')=='downloaded_or_resumed' for r in records),
            'validated_existing_waveforms':sum(r.get('waveform_status')=='validated_existing' for r in records),
            'failed_records':sum(r['status']=='failed' for r in records),
            'not_attempted_records':sum(r['status']=='not_attempted' for r in records),
            'stop_reason':stop_reason,'logical_X_shape':[len(successful),12,1000],
            'status':'complete' if len(successful)==len(cohort) else 'incomplete',
            'records':records,'class_distribution':classes,'patient_fold_consistent':True}
        write_new_json(run_dir/'PTBXL_DOWNLOAD_MANIFEST.json',manifest)
        write_csv_new(run_dir/'PTBXL_CLASS_DISTRIBUTION.csv',classes)
        write_csv_new(run_dir/'PTBXL_PATIENT_FOLD_DISTRIBUTION.csv',folds)
        write_run_report(run_dir/'PTBXL_FULL_EXPORT_REPORT.md',manifest)
        return run_dir,manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata-dir',type=Path,default=DEFAULT_METADATA_DIR)
    parser.add_argument('--waveform-root',type=Path,default=DEFAULT_WAVEFORM_ROOT)
    parser.add_argument('--output-dir',type=Path,default=Path('data/processed/ptbxl_rhythm_v1'))
    parser.add_argument('--dry-run-report',type=Path)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--approve-bulk',action='store_true',help='Use ONLY after explicit approval of Checkpoint A')
    args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    data=load_ptb_metadata(args.metadata_dir)
    if args.execute:
        if not args.approve_bulk:
            parser.error('--execute requires --approve-bulk and prior explicit user approval')
        run_dir,result=run_export(data,approved=True,waveform_root=args.waveform_root,output_dir=args.output_dir,stop_on_error=True)
        print(run_dir)
        print(json.dumps({k:result[k] for k in ('status','selected_records','successful_records','failed_records','not_attempted_records','stop_reason')},indent=2))
    else:
        result=dry_run(data,waveform_root=args.waveform_root)
        if args.dry_run_report:
            write_new_json(args.dry_run_report,result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('files','local_inventory')},indent=2))


if __name__=='__main__':
    main()
