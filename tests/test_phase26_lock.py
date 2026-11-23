"""A consumed final-evaluation claim must stop before source or model access."""
import pytest
from src import mitbih_final_test as final
from src.mitbih_execution_guard import HoldoutAccessGuard


def test_consumed_claim_blocks_preflight_before_checkpoint_or_dataset_io(tmp_path,monkeypatch):
    root=tmp_path/'evaluation';root.mkdir()
    final.claim_once(root/'evaluation_claim.json',{'attempt':'consumed'})
    monkeypatch.setattr(final,'ROOT',root)
    def forbidden(*args,**kwargs):raise AssertionError('No checkpoint/dataset access after consumed claim')
    monkeypatch.setattr(final,'MITBIHExportIndex',forbidden)
    monkeypatch.setattr(final,'sha256_file',forbidden)
    with HoldoutAccessGuard(tmp_path/'guard.json'):
        with pytest.raises(PermissionError,match='already claimed'):final.preflight(root/'another_run')
    assert not (root/'another_run').exists()
