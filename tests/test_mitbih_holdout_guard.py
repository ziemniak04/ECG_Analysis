"""The external regression guard prevents real hold-out I/O, not tmp fixtures."""
import pytest
from examples import pytest_mitbih_holdout_guard as guard


def test_real_heldout_open_is_blocked_before_reader():
    path=next(iter(guard._BLOCKED))
    with pytest.raises(pytest.skip.Exception):guard._open_builtin(path,'rb')
    assert guard._attempts[-1]['path']==str(path.relative_to(guard._ROOT))


def test_guard_keeps_synthetic_same_record_ids_testable(tmp_path):
    path=tmp_path/(guard._TEST_IDS[0]+'.npz')
    with guard._open_builtin(path,'wb') as f:f.write(b'synthetic fixture')
    with guard._open_io(path,'rb') as f:assert f.read()==b'synthetic fixture'


def test_guard_is_pinned_to_all_nine_frozen_record_ids():
    assert set(guard._TEST_IDS)=={'112','203','207','209','212','213','217','228','234'}
    assert len(guard._BLOCKED)==36
