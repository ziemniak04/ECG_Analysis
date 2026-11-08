"""Normalization behavior, frozen train statistics, serialization and local ECG."""
import copy
import numpy as np
import pytest
from src.data_loader import load_recording
from src.preprocessing import (ECGNormalizer, filter_recording, normalize_ecg,
                               zscore_normalize, minmax_normalize)


@pytest.mark.parametrize('shape', [(12,), (12, 1), (12, 2)])
@pytest.mark.parametrize('method', ['zscore', 'minmax'])
def test_independent_channels_shape_and_nonmutation(shape, method):
    x = np.arange(np.prod(shape), dtype=float).reshape(shape)
    if len(shape) == 2 and shape[1] == 2:
        x[:, 1] = x[:, 1] * 100 + 500
    original = x.copy()
    result = normalize_ecg(x, method)
    assert result.shape == shape
    assert not np.shares_memory(result, x)
    np.testing.assert_array_equal(x, original)
    if method == 'zscore':
        np.testing.assert_allclose(result.mean(axis=0), 0, atol=1e-14)
        np.testing.assert_allclose(result.std(axis=0), 1)
    else:
        np.testing.assert_allclose(result.min(axis=0), 0)
        np.testing.assert_allclose(result.max(axis=0), 1)


def test_known_formulas():
    x = np.array([1, 2, 3])
    np.testing.assert_allclose(zscore_normalize(x), (x-2)/np.sqrt(2/3))
    np.testing.assert_allclose(minmax_normalize(x, feature_range=(-1, 1)), [-1, 0, 1])


@pytest.mark.parametrize('method', ['zscore', 'minmax'])
@pytest.mark.parametrize('x', [np.ones(10), np.ones((10, 2)), np.array([4.]),
                              np.array([0., 1e-12, 2e-12])])
def test_constant_and_near_constant(x, method):
    result = normalize_ecg(x, method)
    assert np.isfinite(result).all()
    np.testing.assert_array_equal(result, np.zeros_like(x))


def test_mixed_constant_channels_and_configurable_epsilon():
    x = np.column_stack([np.ones(3), [0, 1, 2]])
    result = zscore_normalize(x)
    np.testing.assert_array_equal(result[:, 0], 0)
    assert result[:, 1].std() == pytest.approx(1)
    np.testing.assert_allclose(minmax_normalize([0, 1e-10], epsilon=1e-12), [0, 1])
    np.testing.assert_array_equal(minmax_normalize([2, 2], feature_range=(-1, 1)), [-1, -1])


@pytest.mark.parametrize('values', [[], np.empty((2,0)), np.zeros((2,2,2)),
                                  [np.nan, 1], [np.inf, 1], [-np.inf], ['1','2'], [1j], [True]])
def test_invalid_values(values):
    with pytest.raises(ValueError):
        normalize_ecg(values)


@pytest.mark.parametrize('kwargs', [{'method':'bad'}, {'epsilon':0}, {'epsilon':np.nan},
                                   {'epsilon':-1}, {'feature_range':(1,1)},
                                   {'feature_range':(2,1)}, {'feature_range':(0,np.inf)},
                                   {'feature_range':(0,)}, {'clip':1}, {'clip':True}])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        ECGNormalizer(**kwargs)


@pytest.mark.parametrize('method', ['zscore', 'minmax'])
def test_training_fit_transform_does_not_refit(method, tmp_path):
    train = np.array([[1., 10.], [3., 20.], [5., 30.]])
    held_out = np.array([[9., 50.], [11., 60.]])
    normalizer = ECGNormalizer(method).fit(train, scope='training', source_ids=['train_record'],
                                          channel_names=['MLII','V5'])
    original_state = normalizer.to_dict()
    result = normalizer.transform(held_out, channel_names=['MLII','V5'])
    if method == 'zscore':
        expected = (held_out-train.mean(axis=0))/train.std(axis=0)
    else:
        expected = (held_out-train.min(axis=0))/np.ptp(train, axis=0)
        assert result.max() > 1  # No automatic clipping of unseen values.
    np.testing.assert_allclose(result, expected)
    assert normalizer.to_dict() == original_state
    path = tmp_path/'normalizer.json'
    normalizer.save(path)
    restored = ECGNormalizer.load(path)
    np.testing.assert_array_equal(restored.transform(held_out, channel_names=['MLII','V5']), result)
    assert restored.to_dict() == original_state
    with pytest.raises(FileExistsError):
        normalizer.save(path)
    changed = normalizer.to_dict()
    changed['offset'][0] = 1000
    assert normalizer.to_dict() == original_state


def test_clip_and_constant_training_channel():
    scaler = ECGNormalizer('minmax', clip=True).fit([0,1], scope='training', source_ids=['t'])
    np.testing.assert_array_equal(scaler.transform([-5, 5]), [0, 1])
    scaler = ECGNormalizer().fit([2,2], scope='training', source_ids=['t'])
    np.testing.assert_array_equal(scaler.transform([1,10]), [0,0])


def test_not_fitted_and_channel_mismatch():
    scaler = ECGNormalizer()
    with pytest.raises(ValueError):
        scaler.transform([1,2])
    with pytest.raises(ValueError):
        scaler.to_dict()
    scaler.fit([[1,2],[3,4]], scope='recording', source_ids=['100'], channel_names=['MLII','V5'])
    for values, names in [([1,2], None), ([[1,2]], None), ([[1,2]], ['V5','MLII'])]:
        with pytest.raises(ValueError):
            scaler.transform(values, channel_names=names)


@pytest.mark.parametrize('kwargs', [{'scope':'global','source_ids':['a']},
                                   {'scope':'training','source_ids':[]},
                                   {'scope':'training','source_ids':'100'},
                                   {'scope':'segment','source_ids':['']},
                                   {'scope':'recording','source_ids':['a'],'channel_names':['a','b']}])
def test_fit_scope_and_provenance(kwargs):
    with pytest.raises(ValueError):
        ECGNormalizer().fit([1,2], **kwargs)


@pytest.mark.parametrize('field,value', [('version',2), ('scale',[-1]), ('offset',[np.nan]),
                                       ('scale',[1,2]), ('sample_count',0), ('scope','global'),
                                       ('source_ids',[]), ('channel_names',['a','b'])])
def test_invalid_saved_state(field, value):
    state = ECGNormalizer().fit([1,2], scope='training', source_ids=['train']).to_dict()
    state[field] = value
    with pytest.raises(ValueError):
        ECGNormalizer.from_dict(state)


def test_extreme_numeric_values_fail_safely():
    with pytest.raises(ValueError, match='numeric range'):
        zscore_normalize([1e308, -1e308])


@pytest.mark.parametrize('rid', ['100','101','200'])
@pytest.mark.parametrize('method', ['zscore','minmax'])
def test_local_filtered_ecg(rid, method):
    record = filter_recording(load_recording(rid))
    before = record.signals.copy()
    scaler = ECGNormalizer(method).fit(record.signals, scope='recording', source_ids=[rid],
                                       channel_names=record.channel_names)
    result = scaler.transform(record.signals, channel_names=record.channel_names)
    assert result.shape == before.shape
    assert np.isfinite(result).all()
    np.testing.assert_array_equal(record.signals, before)
    if method == 'zscore':
        np.testing.assert_allclose(result.mean(axis=0), 0, atol=1e-12)
        np.testing.assert_allclose(result.std(axis=0), 1, atol=1e-12)
    else:
        np.testing.assert_allclose(result.min(axis=0), 0)
        np.testing.assert_allclose(result.max(axis=0), 1)
