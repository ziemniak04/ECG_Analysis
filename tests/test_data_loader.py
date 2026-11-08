"""Local integration tests and small malformed-input checks."""
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
from src.data_loader import (DEFAULT_DATA_DIR, inspect_recordings, list_recordings,
                             load_recording, locate_data_dir)


def test_inventory():
    ids = list_recordings()
    assert len(ids) == 48
    assert "100" in ids
    assert all(not missing for missing in inspect_recordings().values())


def test_load_record_100():
    data = load_recording("100")
    assert data.record_id == "100"
    assert data.signals.shape == (650000, 2)
    assert data.sampling_frequency == 360
    assert data.channel_names == ("MLII", "V5")
    assert data.units == ("mV", "mV")
    assert np.isfinite(data.signals).all()
    assert len(data.annotation_positions) == len(data.annotation_symbols) == len(data.annotation_aux_notes)
    assert len(data.annotation_positions) > 2000
    assert data.annotation_positions[0] == 18
    assert data.annotation_symbols[0] == "+"
    assert "N" in data.annotation_symbols
    assert np.all(np.diff(data.annotation_positions) >= 0)


@pytest.mark.parametrize("record_id", list_recordings())
def test_every_local_recording(record_id):
    data = load_recording(record_id)
    assert data.record_id == record_id
    assert data.signals.shape[1] == len(data.channel_names)
    assert len(data.annotation_positions) > 0


def test_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        locate_data_dir(tmp_path / "missing")
    with pytest.raises(FileNotFoundError):
        locate_data_dir(tmp_path)


def test_incomplete_recording(tmp_path):
    (tmp_path / "100.hea").touch()
    assert list_recordings(tmp_path) == []
    assert inspect_recordings(tmp_path)["100"] == ("dat", "atr")
    with pytest.raises(FileNotFoundError, match="100.dat"):
        load_recording("100", tmp_path)


def test_ambiguous_directories(tmp_path):
    for name in ("a", "b"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "100.hea").touch()
    with pytest.raises(ValueError, match="Multiple"):
        locate_data_dir(tmp_path)


@pytest.mark.parametrize("record_id", ["../100", "100.hea", "", "abc"])
def test_invalid_id(record_id):
    with pytest.raises(ValueError):
        load_recording(record_id)


def test_corrupt_file(tmp_path):
    for ext in ("hea", "dat", "atr"):
        (tmp_path / f"100.{ext}").write_text("invalid")
    with pytest.raises(ValueError, match="Cannot read"):
        load_recording("100", tmp_path)


def test_invalid_dimensions():
    from types import SimpleNamespace
    fake = SimpleNamespace(p_signal=np.zeros(10), sig_len=10, n_sig=1)
    annotation = SimpleNamespace(sample=[], symbol=[])
    with patch("src.data_loader.wfdb.rdrecord", return_value=fake), patch("src.data_loader.wfdb.rdann", return_value=annotation):
        with pytest.raises(ValueError, match="dimensions"):
            load_recording("100")
