"""Frequency-selective behavior, invalid inputs, and local ECG integration."""
import numpy as np
import pytest
from src.data_loader import load_recording
from src.preprocessing import (highpass_filter, lowpass_filter, notch_filter,
                               filter_ecg, filter_recording, powerline_diagnostics)

FS = 360


def amplitude(values, frequency):
    # Ignore padding transients when checking the steady-state frequency response.
    trimmed = values[5 * FS:-5 * FS]
    t = np.arange(5 * FS, len(values) - 5 * FS) / FS
    return abs(2 * np.mean(trimmed * np.exp(-2j * np.pi * frequency * t)))


def tones():
    t = np.arange(30 * FS) / FS
    return sum(np.sin(2 * np.pi * f * t) for f in (0.1, 10, 90))


def test_highpass_rejects_baseline_preserves_passband():
    result = highpass_filter(tones(), FS)
    assert amplitude(result, 0.1) < 0.05
    assert 0.95 < amplitude(result, 10) < 1.05


def test_lowpass_rejects_noise_preserves_passband():
    result = lowpass_filter(tones(), FS)
    assert amplitude(result, 90) < 0.02
    assert 0.95 < amplitude(result, 10) < 1.05


@pytest.mark.parametrize('frequency', [50, 60])
def test_notch_suppression(frequency):
    t = np.arange(30 * FS) / FS
    values = np.sin(2*np.pi*frequency*t) + np.sin(2*np.pi*10*t)
    result = notch_filter(values, FS, frequency)
    assert amplitude(result, frequency) < 0.02
    assert 0.95 < amplitude(result, 10) < 1.05


def test_zero_phase_passband():
    t = np.arange(30 * FS) / FS
    values = np.sin(2*np.pi*10*t)
    result = filter_ecg(values, FS)
    interior = slice(5*FS, -5*FS)
    assert np.corrcoef(values[interior], result[interior])[0, 1] > 0.9999


@pytest.mark.parametrize('fn', [highpass_filter, lowpass_filter, notch_filter, filter_ecg])
@pytest.mark.parametrize('channels', [None, 1, 2])
def test_shape_finite_and_nonmutation(fn, channels):
    values = tones()
    if channels is not None:
        values = np.tile(values[:, None], (1, channels))
    original = values.copy()
    result = fn(values, FS)
    assert result.shape == values.shape
    assert result.dtype == np.float64
    assert np.isfinite(result).all()
    np.testing.assert_array_equal(values, original)
    if channels == 2:
        np.testing.assert_allclose(result[:, 0], fn(values[:, 0], FS))


@pytest.mark.parametrize('fn', [highpass_filter, lowpass_filter, notch_filter, filter_ecg])
def test_constant_and_short(fn):
    result = fn(np.ones((2000, 2)), FS)
    assert np.isfinite(result).all()
    if fn in (highpass_filter, filter_ecg):
        assert np.max(np.abs(result)) < 1e-10
    else:
        np.testing.assert_allclose(result, 1, atol=1e-10)
    with pytest.raises(ValueError, match='too short'):
        fn(np.ones(3), FS)


@pytest.mark.parametrize('values', [[], [1, np.nan], [np.inf, 0], [[[]]], ['a', 'b'], [1+2j]])
def test_invalid_signal(values):
    with pytest.raises(ValueError):
        filter_ecg(values, FS)


@pytest.mark.parametrize('fs', [0, -1, np.nan, np.inf, True, 'invalid'])
def test_invalid_fs(fs):
    with pytest.raises(ValueError):
        filter_ecg(tones(), fs)


@pytest.mark.parametrize('cutoff', [0, -1, FS/2, FS, np.inf, np.nan, True, '0.5'])
@pytest.mark.parametrize('fn', [highpass_filter, lowpass_filter])
def test_invalid_cutoff(fn, cutoff):
    with pytest.raises(ValueError):
        fn(tones(), FS, cutoff)


@pytest.mark.parametrize('order', [0, -1, 1.5, True])
def test_invalid_order(order):
    with pytest.raises(ValueError):
        filter_ecg(tones(), FS, order=order)


@pytest.mark.parametrize('kwargs', [{'highpass_hz': 45, 'lowpass_hz': 0.5}, {'notch_hz': 55},
                                   {'notch_hz': 50, 'quality_factor': 0},
                                   {'notch_hz': 50, 'quality_factor': np.nan}])
def test_invalid_pipeline(kwargs):
    with pytest.raises(ValueError):
        filter_ecg(tones(), FS, **kwargs)


def test_pipeline_order_and_optional_stages():
    values = tones()
    expected = lowpass_filter(highpass_filter(notch_filter(values, FS, 50), FS), FS)
    np.testing.assert_allclose(filter_ecg(values, FS, notch_hz=50), expected)
    np.testing.assert_allclose(filter_ecg(values, FS), lowpass_filter(highpass_filter(values, FS), FS))
    result = filter_ecg(values, FS, highpass_hz=None, lowpass_hz=None)
    np.testing.assert_array_equal(result, values)
    assert not np.shares_memory(result, values)


def test_diagnostic_detects_added_line_and_constant():
    rng = np.random.default_rng(42)
    values = rng.normal(0, 0.1, 10*FS)
    base = powerline_diagnostics(values, FS)
    values += np.sin(2*np.pi*50*np.arange(len(values))/FS)
    line = powerline_diagnostics(values, FS)
    assert line['peak_to_background_ratio'][0] > 100 * base['peak_to_background_ratio'][0]
    assert powerline_diagnostics(np.ones(10*FS), FS)['peak_to_background_ratio'] == [0.0]
    with pytest.raises(ValueError):
        powerline_diagnostics(np.ones(20), FS)


@pytest.mark.parametrize('rid', ['100', '101', '200'])
def test_real_record_filtering_and_metadata(rid):
    raw = load_recording(rid)
    original = raw.signals.copy()
    filtered = filter_recording(raw)
    assert filtered.signals.shape == raw.signals.shape
    assert np.isfinite(filtered.signals).all()
    assert filtered.sampling_frequency == raw.sampling_frequency
    assert filtered.record_id == raw.record_id
    assert filtered.channel_names == raw.channel_names
    assert filtered.units == raw.units
    np.testing.assert_array_equal(filtered.annotation_positions, raw.annotation_positions)
    assert filtered.annotation_symbols == raw.annotation_symbols
    assert filtered.annotation_aux_notes == raw.annotation_aux_notes
    np.testing.assert_array_equal(raw.signals, original)
