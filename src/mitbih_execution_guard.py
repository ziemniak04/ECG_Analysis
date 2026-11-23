"""Fail-closed runtime audit-hook guard for real MIT-BIH held-out files.

Unlike the pytest plugin this raises PermissionError, never skips execution.
Installed before preflight/training/evaluation; original guard/tests stay unchanged.
Python audit hooks are disabled by the context on exit (cannot be unregistered).
"""
from contextlib import AbstractContextManager
import hashlib
import json
import os
from pathlib import Path
import sys

_ACTIVE=[]
SPLIT_SHA256='8c5cc28d8dcae170fd222432d9f247fd19207ba01be60763241da4b7cae61831'


class HoldoutAccessGuard(AbstractContextManager):
    def __init__(self,log_path,root=None):
        self.root=(Path(__file__).resolve().parents[1] if root is None else Path(root).resolve())
        split_path=self.root/'configs/mitbih_patient_split_v1.json'
        if hashlib.sha256(split_path.read_bytes()).hexdigest()!=SPLIT_SHA256:raise ValueError('Frozen split differs')
        self.test_ids=json.loads(split_path.read_text())['summary']['test']['record_ids']
        raw=self.root/'data/mitdb/physionet.org/files/mitdb/1.0.0'
        ecg=self.root/'data/processed/ecg_dataset_v1/recordings'
        rr=self.root/'data/processed/mitbih_rr_context_v1/recordings'
        self.blocked={str((raw/f'{r}.{ext}').resolve()) for r in self.test_ids for ext in ('dat','atr','hea')}
        self.blocked.update(str((ecg/f'{r}.npz').resolve()) for r in self.test_ids)
        self.blocked.update(str((rr/f'{r}_rr.npz').resolve()) for r in self.test_ids)
        self.log_path=Path(log_path)
        if self.log_path.exists():raise FileExistsError('Guard log must be new')
        self.active=False;self.probing=False;self.expected=[];self.unexpected=[]

    def _hook(self,event,args):
        if not self.active or event!='open':return
        try:
            value=os.fsdecode(os.fspath(args[0]))
            # Limit expensive path resolution to real data paths / potential aliases.
            path=str(Path(value).resolve())
        except (TypeError,ValueError,OSError):return
        if path in self.blocked:
            row=dict(path=path,operation='open denied before reader',probe=self.probing)
            (self.expected if self.probing else self.unexpected).append(row)
            raise PermissionError('Real MIT-BIH final-test file is locked by execution guard')

    def __enter__(self):
        if self.active:raise RuntimeError('Guard already active')
        sys.addaudithook(self._hook);self.active=True;_ACTIVE.append(self)
        return self

    def probe(self):
        self.probing=True
        try:
            for path in sorted(self.blocked):
                try:
                    fd=os.open(path,os.O_RDONLY)
                    os.close(fd)
                    raise RuntimeError('Isolation probe unexpectedly opened a held-out file')
                except PermissionError:pass
        finally:self.probing=False
        if len(self.expected)!=len(self.blocked):raise RuntimeError('Incomplete isolation probe')
        return dict(blocked_paths=len(self.blocked),expected_denials=len(self.expected),actual_reads=0)

    def __exit__(self,kind,value,tb):
        self.active=False
        _ACTIVE.remove(self)
        self.log_path.parent.mkdir(parents=True,exist_ok=True)
        with self.log_path.open('x') as f:
            json.dump(dict(schema='mitbih-execution-guard-v1',test_record_ids=self.test_ids,
                expected_probe_denials=self.expected,unexpected_denials=self.unexpected,
                unexpected_denial_count=len(self.unexpected),guarded_heldout_open_successes=0,
                exit_exception=None if kind is None else kind.__name__,
                scope='Python open audit events on real raw dat/atr/hea, original NPZ, deferred RR NPZ; not an OS sandbox',
                historical_24A_QC_incident_acknowledged=True),f,indent=2)
        return False


def require_guard():
    if not _ACTIVE or not all(g.active for g in _ACTIVE):raise PermissionError('An active hold-out execution guard is required')
    return _ACTIVE[-1]
