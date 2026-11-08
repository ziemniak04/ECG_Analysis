"""Time/annotation alignment, defaults, statistics and comparison artifacts."""
import csv
from dataclasses import replace
import numpy as np
import pytest
from PIL import Image
from src.data_loader import ECGRecording, load_recording
from src.preprocessing import filter_recording, normalize_ecg
from src.preprocessing_comparison import (STAGES, compare_recording, comparison_statistics,
    plot_preprocessing_comparison, save_comparison_figure, save_comparison_statistics)
import matplotlib.pyplot as plt


@pytest.fixture
def record():
    fs = 360
    t = np.arange(30*fs)/fs
    values = np.column_stack([np.sin(2*np.pi*2*t)+0.3*np.sin(2*np.pi*0.1*t),
                              2*np.sin(2*np.pi*3*t)+0.1*np.sin(2*np.pi*70*t)])
    positions = np.array([0, 3600, 3610, 4000, 7199, 7200])
    symbols = ('N', 'N', 'V', '+', 'A', 'N')
    return ECGRecording('synthetic', values, fs, ('MLII','V5'), ('mV','mV'),
                        positions, symbols, ('',)*len(positions))


def test_default_pipeline_shapes_time_and_nonmutation(record):
    original = record.signals.copy()
    comparison = compare_recording(record)
    assert comparison.start_sample == 3600
    assert comparison.stop_sample == 7200
    assert comparison.sampling_frequency == record.sampling_frequency
    np.testing.assert_array_equal(comparison.time, np.arange(3600,7200)/360)
    assert all(values.shape == (3600,2) for values in comparison.stages.values())
    filtered = filter_recording(record)
    normalized = normalize_ecg(filtered.signals, source_ids=[record.record_id], channel_names=record.channel_names)
    for stage, expected in zip(STAGES, (record.signals, filtered.signals, normalized)):
        np.testing.assert_array_equal(comparison.stages[stage], expected[3600:7200])
    np.testing.assert_array_equal(record.signals, original)
    # Normalization cannot relocate a maximum for these nonconstant channels.
    np.testing.assert_array_equal(comparison.stages['filtered'].argmax(axis=0),
                                  comparison.stages['filtered_zscore'].argmax(axis=0))


def test_half_open_annotation_interval_and_nonbeats_excluded(record):
    comparison = compare_recording(record)
    np.testing.assert_array_equal(comparison.annotation_positions, [3600,3610,7199])
    assert comparison.annotation_symbols == ('N','V','A')
    np.testing.assert_array_equal(record.annotation_positions, [0,3600,3610,4000,7199,7200])


@pytest.mark.parametrize('channels', [1,2])
def test_plot_alignment_labels_markers_and_png(record, channels, tmp_path):
    record = replace(record, signals=record.signals[:,:channels],
                     channel_names=record.channel_names[:channels], units=record.units[:channels])
    comparison = compare_recording(record)
    fig = plot_preprocessing_comparison(comparison)
    try:
        assert len(fig.axes) == channels*3
        for index, ax in enumerate(fig.axes):
            channel, column = divmod(index,3)
            np.testing.assert_array_equal(ax.lines[0].get_xdata(), comparison.time)
            np.testing.assert_array_equal(ax.lines[0].get_ydata(), comparison.stages[STAGES[column]][:,channel])
            offsets = ax.collections[0].get_offsets()
            np.testing.assert_array_equal(offsets[:,0], comparison.annotation_positions/360)
            local = comparison.annotation_positions - comparison.start_sample
            np.testing.assert_array_equal(offsets[:,1], comparison.stages[STAGES[column]][local,channel])
            assert [txt.get_text() for txt in ax.texts] == list(comparison.annotation_symbols)
            assert ax.get_xlim() == (10,20)
            assert ('dimensionless' if column == 2 else 'mV') in ax.get_ylabel()
            assert record.channel_names[channel] in ax.get_ylabel()
        assert '360 Hz' in fig._suptitle.get_text()
        path = save_comparison_figure(fig, tmp_path/'comparison.png')
        with Image.open(path) as png:
            assert png.format == 'PNG'
            assert png.width == 5400
            assert png.height == 1200*channels
            assert png.info['dpi'][0] == pytest.approx(300, abs=0.1)
        with pytest.raises(FileExistsError):
            save_comparison_figure(fig, path)
    finally:
        plt.close(fig)


def test_no_beat_annotations(record):
    record = replace(record, annotation_positions=np.array([],dtype=int),
                     annotation_symbols=(), annotation_aux_notes=())
    fig = plot_preprocessing_comparison(compare_recording(record))
    try:
        assert all(len(ax.texts) == 0 for ax in fig.axes)
    finally:
        plt.close(fig)


def test_statistics_match_displayed_samples_and_csv(record, tmp_path):
    comparison = compare_recording(record)
    rows = comparison_statistics(comparison)
    assert len(rows) == 6
    for row in rows:
        values = comparison.stages[row['stage']][:,row['channel_index']]
        for name, expected in [('mean', values.mean()), ('std', values.std(ddof=0)),
                               ('min', values.min()), ('max', values.max())]:
            assert row[name] == pytest.approx(expected)
        assert row['sample_count'] == 3600
        assert row['statistics_scope'] == 'displayed_fragment'
        assert row['normalization_scope'] == ('recording_per_channel' if row['stage']=='filtered_zscore' else 'none')
    path = save_comparison_statistics(rows, tmp_path/'stats.csv')
    with path.open() as source:
        saved = list(csv.DictReader(source))
    assert len(saved) == 6
    assert saved[0]['record_id'] == record.record_id
    assert saved[0]['mean'] == str(rows[0]['mean'])
    with pytest.raises(FileExistsError):
        save_comparison_statistics(rows, path)
    with pytest.raises(ValueError):
        save_comparison_statistics([], tmp_path/'empty.csv')


@pytest.mark.parametrize('start,duration', [(-1,10),(0,0),(0,-1),(0,np.nan),
                                          (np.inf,10),(25,10),(10,0.00001),('10',10)])
def test_invalid_interval(record, start, duration):
    with pytest.raises(ValueError):
        compare_recording(record,start,duration)


@pytest.mark.parametrize('rid', ['100','101','200'])
def test_local_record_comparison(rid):
    comparison = compare_recording(load_recording(rid))
    assert comparison.record_id == rid
    assert all(x.shape == (3600,2) and np.isfinite(x).all() for x in comparison.stages.values())
    assert len(comparison.time) == 3600
    assert len(comparison.annotation_positions) > 0
    assert len(comparison_statistics(comparison)) == 6
