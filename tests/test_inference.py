"""The app must preprocess exactly like the training data was made.

Uses a made-up signal.
"""
import numpy as np
import pytest

from src.ml import inference as inf
from src.preprocessing import ECGNormalizer, filter_ecg
from src.segmentation import segment_heartbeats

def fake_signal(n=7200, fs=360, leads=2):
    rng = np.random.default_rng(0)
    t = np.arange(n) / fs
    base = np.sin(2 * np.pi * 1.2 * t) + 0.3 * np.sin(2 * np.pi * 0.2 * t)
    return np.stack([base * (k + 1) + rng.normal(0, 0.05, n) for k in range(leads)], axis=1)

def test_prepare_matches_training_pipeline():
    sig = fake_signal()
    filtered = filter_ecg(sig, 360)
    norm = ECGNormalizer().fit(filtered, scope="recording", source_ids=["x"],
                               channel_names=("a", "b")).transform(filtered, channel_names=("a", "b"))
    f2, n2 = inf.prepare(sig, 360)
    assert np.array_equal(filtered, f2)
    assert np.array_equal(norm, n2)

def test_beat_windows_match_segmentation_code():
    _, norm = inf.prepare(fake_signal(), 360)
    peaks = np.array([500, 900, 1400, 2000])
    ours = np.stack([norm[p - inf.BEFORE:p + inf.AFTER].T for p in peaks])
    theirs = segment_heartbeats(norm, 360, peaks, np.array(["N"] * 4), record_id="t",
                                channel_names=("a", "b"), units=("dimensionless",) * 2).X
    assert np.array_equal(ours, theirs)

def test_order_leads():
    sig = np.arange(24).reshape(2, 12).astype(float)
    names = ["II", "I", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    out, note = inf.order_leads(sig, names)
    assert np.array_equal(out[:, 0], sig[:, 1]) and note
    with pytest.raises(ValueError):
        inf.order_leads(sig, ["x"] * 12)

def test_csv_reading():
    data = b"time,I,II\n0,1.0,2.0\n0.01,1.5,2.5\n"
    out = inf.read_csv_file(data)
    assert out["signal"].shape == (2, 2) and out["names"] == ["I", "II"]
    with pytest.raises(ValueError):
        inf.read_csv_file(b"I,II\n1,abc\n")