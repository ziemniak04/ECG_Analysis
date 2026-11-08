"""Expert alignment, boundaries, metadata and full-record ECG integration."""
from dataclasses import replace
import numpy as np
import pytest
from src.data_loader import ECGRecording, load_recording
from src.preprocessing import ECGNormalizer, filter_recording
from src.segmentation import window_samples, segment_heartbeats, segment_recording
from src.visualization import BEAT_SYMBOLS


def extract(values, positions, symbols, **kwargs):
    return segment_heartbeats(values, 360, positions, symbols, record_id='synthetic', **kwargs)


def test_default_window_and_exact_alignment():
    x = np.arange(1000, dtype=float)
    original = x.copy()
    result = extract(x, [300,600], ['N','V'])
    assert window_samples(360) == (90,162)
    assert result.X.shape == (2,1,252)
    np.testing.assert_array_equal(result.X[0,0], x[210:462])
    np.testing.assert_array_equal(result.X[:,0,90], x[[300,600]])
    np.testing.assert_array_equal(x, original)
    assert not np.shares_memory(x, result.X)


def test_rounding_and_custom_window():
    assert window_samples(250) == (63,113)
    assert window_samples(360,0.1,0.2) == (36,72)
    result = extract(np.arange(1000), [300], ['N'], before_seconds=0.1, after_seconds=0.2)
    assert result.X.shape == (1,1,108)
    assert result.X[0,0,36] == 300


def test_exact_boundaries_and_skip_report():
    result = extract(np.arange(1000), [0,89,90,838,839,999], ['N']*6)
    np.testing.assert_array_equal(result.annotation_positions,[90,838])
    assert result.X[0,0,0] == 0
    assert result.X[-1,0,-1] == 999
    assert result.report['skipped_boundary'] == 4
    assert result.report['skipped_left'] == result.report['skipped_right'] == 2
    assert result.report['skipped_boundary_annotation_indices'] == [0,1,4,5]


def test_window_larger_than_signal_reports_both_edges_once():
    result = extract(np.zeros(10), [5], ['N'])
    assert result.X.shape == (0,1,252)
    assert result.report['skipped_boundary'] == 1
    assert result.report['skipped_left'] == result.report['skipped_right'] == 1


@pytest.mark.parametrize('channels',[1,2,3])
def test_multichannel_metadata_and_symbols(channels):
    values = np.arange(1000*channels).reshape(1000,channels)
    names = tuple(f'ch{i}' for i in range(channels))
    result = extract(values,[300,400],['A','/'],channel_names=names,units=('mV',)*channels,
                     annotation_aux_notes=['note1','note2'],patient_group_id='known_group')
    assert result.X.shape == (2,channels,252)
    np.testing.assert_array_equal(result.X[0],values[210:462].T)
    np.testing.assert_array_equal(result.annotation_symbols,['A','/'])
    np.testing.assert_array_equal(result.annotation_aux_notes,['note1','note2'])
    np.testing.assert_array_equal(result.record_ids,['synthetic']*2)
    np.testing.assert_array_equal(result.patient_group_ids,['known_group']*2)
    assert result.channel_names == names
    assert result.sampling_frequency == 360


def test_excludes_nonbeat_markers_but_keeps_all_beat_symbols():
    symbols = sorted(BEAT_SYMBOLS)+['+','~','|','p','t','?','!','[',']','x','^','(',')']
    positions = np.arange(len(symbols))+300
    result = extract(np.zeros(1000),positions,symbols)
    assert result.annotation_symbols.tolist() == sorted(BEAT_SYMBOLS)
    assert result.report['excluded_nonheartbeat'] == 13
    assert result.report['excluded_symbol_counts']['+'] == 1
    assert result.report['total_annotations'] == len(symbols)
    assert result.patient_group_ids.tolist() == ['']*len(BEAT_SYMBOLS)


@pytest.mark.parametrize('positions,symbols',[([],[]),([300],['+']),([0],['N'])])
def test_no_valid_beats(positions,symbols):
    result = extract(np.ones((1000,2)),positions,symbols)
    assert result.X.shape == (0,2,252)
    assert result.annotation_positions.shape == (0,)
    assert result.record_ids.shape == (0,)
    assert np.isfinite(result.X).all()


def test_duplicate_positions_keep_original_annotation_indices():
    result = extract(np.zeros(1000),[300,300,400],['N','V','+'])
    assert result.annotation_indices.tolist() == [0,1]
    assert result.annotation_positions.tolist() == [300,300]


@pytest.mark.parametrize('positions,symbols',[([-1],['N']),([1000],['N']),([1.5],['N']),
    ([400,300],['N','V']),([300],[]),([[300]],['N']),([300],['NN'])])
def test_invalid_annotations(positions,symbols):
    with pytest.raises(ValueError):
        extract(np.zeros(1000),positions,symbols)


@pytest.mark.parametrize('values',[[np.nan]*1000,[np.inf]*1000,[],np.zeros((2,2,2))])
def test_invalid_signals(values):
    with pytest.raises(ValueError):
        extract(values,[],[])


@pytest.mark.parametrize('fs,before,after',[(0,.25,.45),(360,0,.45),(360,.25,-1),
    (360,np.nan,.45),(360,.25,np.inf),(360,1e-6,.45),(360,1e308,.45)])
def test_invalid_window(fs,before,after):
    with pytest.raises(ValueError):
        window_samples(fs,before,after)


def test_supplied_train_scaler_is_not_refitted():
    fs=360
    x=np.column_stack([np.sin(np.arange(2000)/10),np.cos(np.arange(2000)/20)])
    record=ECGRecording('synthetic',x,fs,('a','b'),('mV','mV'),np.array([300]),('N',),('',))
    scaler=ECGNormalizer().fit(x*2,scope='training',source_ids=['train_only'],channel_names=record.channel_names)
    state=scaler.to_dict()
    result=segment_recording(record,normalizer=scaler)
    expected=scaler.transform(filter_recording(record).signals,channel_names=record.channel_names)
    np.testing.assert_array_equal(result.X[0],expected[210:462].T)
    assert scaler.to_dict() == state
    assert result.configuration['normalization']['scope'] == 'training'


@pytest.mark.parametrize('rid',['100','101','200'])
def test_local_record_pipeline(rid):
    record=load_recording(rid)
    original=record.signals.copy()
    result=segment_recording(record)
    assert result.X.shape[1:] == (2,252)
    assert len(result.X)>1000
    assert np.isfinite(result.X).all()
    assert np.all(result.record_ids == rid)
    np.testing.assert_array_equal(result.annotation_positions,record.annotation_positions[result.annotation_indices])
    assert result.annotation_symbols.tolist() == [record.annotation_symbols[i] for i in result.annotation_indices]
    assert all(s in BEAT_SYMBOLS for s in result.annotation_symbols)
    assert result.report['total_annotations'] == (len(result.X)+result.report['skipped_boundary']+
                                                  result.report['excluded_nonheartbeat'])
    assert result.configuration['normalization']['scope'] == 'recording'
    assert result.configuration['normalization']['source_ids'] == [rid]
    assert result.units == ('dimensionless',)*2
    np.testing.assert_array_equal(record.signals,original)


def test_single_channel_record_and_configuration():
    x=np.sin(np.arange(2000)/10)
    record=ECGRecording('single',x,360,('MLII',),('mV',),np.array([300]),('N',),('',))
    result=segment_recording(record,patient_group_id='provided_patient')
    assert result.X.shape == (1,1,252)
    assert result.patient_group_ids.tolist() == ['provided_patient']
    assert result.configuration['filtering'] == {'highpass_hz':0.5,'lowpass_hz':45.0,
        'order':4,'notch_hz':None,'quality_factor':30.0}
    assert result.configuration['patient_group_source'] == 'caller_supplied'
    assert result.configuration['normalization']['channel_names'] == ['MLII']


def test_segment_plot_keeps_expert_zero_alignment():
    from examples.segment_ecg import plot_example_segments
    import matplotlib.pyplot as plt
    result=extract(np.column_stack([np.arange(1000),np.arange(1000)*2]),[300,600],['N','V'])
    fig=plot_example_segments(result,[0,1])
    try:
        assert len(fig.axes) == 4
        expected=(np.arange(252)-90)/360
        for index,ax in enumerate(fig.axes):
            channel,column=divmod(index,2)
            np.testing.assert_array_equal(ax.lines[0].get_xdata(),expected)
            np.testing.assert_array_equal(ax.lines[0].get_ydata(),result.X[column,channel])
            np.testing.assert_array_equal(ax.collections[0].get_offsets(),[[0,result.X[column,channel,90]]])
    finally:
        plt.close(fig)
