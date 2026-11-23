"""External pytest guard: skip legacy QC that opens real held-out MIT-BIH files.

Does not edit production loaders, prior tests, datasets or split locks. Synthetic
fixtures with the same record numbers in tmp_path remain fully testable.
"""
import builtins
import functools
import io
import json
import os
from pathlib import Path
import pytest

_ROOT=Path(__file__).resolve().parents[1]
_split=json.loads((_ROOT/'configs/mitbih_patient_split_v1.json').read_text())
_TEST_IDS=_split['summary']['test']['record_ids']
_RAW=_ROOT/'data/mitdb/physionet.org/files/mitdb/1.0.0'
_PROCESSED=_ROOT/'data/processed/ecg_dataset_v1/recordings'
_BLOCKED={(_RAW/f'{rid}.{ext}').resolve() for rid in _TEST_IDS for ext in ('dat','atr','hea')}
_BLOCKED.update((_PROCESSED/f'{rid}.npz').resolve() for rid in _TEST_IDS)
_original_builtin=builtins.open
_original_io=io.open
_attempts=[]


def _check(file):
    try:path=Path(os.fsdecode(os.fspath(file))).resolve()
    except (TypeError,ValueError,OSError):return
    if path in _BLOCKED:
        _attempts.append(dict(path=str(path.relative_to(_ROOT)),test=os.environ.get('PYTEST_CURRENT_TEST','collection')))
        pytest.skip('Real MIT-BIH held-out files locked: legacy all-record QC deferred; synthetic fixtures remain enabled',allow_module_level=True)


@functools.wraps(_original_builtin)
def _open_builtin(file,*args,**kwargs):
    _check(file)
    return _original_builtin(file,*args,**kwargs)


@functools.wraps(_original_io)
def _open_io(file,*args,**kwargs):
    _check(file)
    return _original_io(file,*args,**kwargs)


def pytest_configure(config):
    builtins.open=_open_builtin;io.open=_open_io


def pytest_sessionfinish(session,exitstatus):
    path=os.environ.get('ECG_HOLDOUT_GUARD_LOG')
    if path:
        with _original_builtin(path,'x') as f:
            json.dump(dict(schema='mitbih-pytest-holdout-guard-v1',exit_status=exitstatus,
                blocked_record_ids=_TEST_IDS,blocked_file_attempts=_attempts,
                actual_heldout_reads_through_guarded_open=0,scope='Real raw dat/atr/hea and processed NPZ; tmp fixtures unaffected'),f,indent=2)


def pytest_unconfigure(config):
    builtins.open=_original_builtin;io.open=_original_io
