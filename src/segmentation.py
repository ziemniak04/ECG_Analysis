"""Expert-annotation heartbeat windows, without class mapping or dataset splitting."""
from dataclasses import dataclass
from collections import Counter
import inspect
import numpy as np
from .data_loader import ECGRecording
from .preprocessing import _input, _positive, filter_ecg, filter_recording, ECGNormalizer
from .visualization import BEAT_SYMBOLS


@dataclass(frozen=True)
class SegmentedHeartbeats:
    """In-memory [beats, channels, samples] data with original annotation metadata.

    Empty patient_group_ids mean unknown, never an inferred patient identity.
    annotation_indices refer to the original .atr annotation array, not samples.
    """
    X: np.ndarray
    record_ids: np.ndarray
    patient_group_ids: np.ndarray
    annotation_positions: np.ndarray
    annotation_indices: np.ndarray
    annotation_symbols: np.ndarray
    annotation_aux_notes: np.ndarray
    sampling_frequency: float
    channel_names: tuple[str, ...]
    units: tuple[str, ...]
    before_samples: int
    after_samples: int
    report: dict
    configuration: dict


def window_samples(sampling_frequency, before_seconds=0.25, after_seconds=0.45):
    """Round each duration independently to nearest sample, half upwards.

    Extract [annotation-before_samples, annotation+after_samples), end exclusive.
    The annotation is at index before_samples; length is their sum, without +1.
    """
    fs = _positive(sampling_frequency, "sampling_frequency")
    counts = []
    for name, value in (("before_seconds", before_seconds), ("after_seconds", after_seconds)):
        duration = _positive(value, name)
        scaled = duration * fs
        if not np.isfinite(scaled) or scaled > np.iinfo(np.int64).max // 4:
            raise ValueError("Window duration exceeds supported sample range")
        count = int(np.floor(scaled + 0.5))
        if count < 1:
            raise ValueError(f"{name} rounds to zero samples")
        counts.append(count)
    return tuple(counts)


def segment_heartbeats(values, sampling_frequency, annotation_positions, annotation_symbols, *,
                       record_id, before_seconds=0.25, after_seconds=0.45,
                       channel_names=None, units=None, annotation_aux_notes=None, patient_group_id=None):
    """Slice supplied ECG as-is; filtering/normalization are NOT repeated here.

    Input: [samples] or [samples, channels]. Only known heartbeat symbols qualify.
    Incomplete windows are skipped, without padding, clipping or R-peak searches.
    Invalid annotations/data cause ValueError rather than silent corrections.
    """
    values, fs = _input(values, sampling_frequency)
    if values.ndim == 1:
        values = values[:, None]
    channels = values.shape[1]
    before, after = window_samples(fs, before_seconds, after_seconds)
    if not isinstance(record_id, str) or not record_id:
        raise ValueError("record_id must be a nonempty string")
    if patient_group_id is not None and (not isinstance(patient_group_id, str) or not patient_group_id):
        raise ValueError("patient_group_id must be a nonempty string or None")
    names = tuple(f"channel_{i}" for i in range(channels)) if channel_names is None else tuple(channel_names)
    signal_units = ("unspecified",) * channels if units is None else tuple(units)
    if (len(names) != channels or len(signal_units) != channels
            or any(not isinstance(n, str) or not n for n in names + signal_units)):
        raise ValueError("Channel names and units must match channel count")
    positions = np.asarray(annotation_positions)
    if positions.ndim != 1 or (positions.size and positions.dtype.kind not in "iu"):
        raise ValueError("Annotation positions must be a 1D integer array")
    if isinstance(annotation_symbols, str):
        raise ValueError("Annotation symbols must be a sequence, not one string")
    symbols = tuple(annotation_symbols)
    notes = ("",) * len(symbols) if annotation_aux_notes is None else tuple(annotation_aux_notes)
    if len(positions) != len(symbols) or len(notes) != len(symbols):
        raise ValueError("Annotation positions, symbols and notes must have matching lengths")
    if any(not isinstance(s, str) or len(s) != 1 for s in symbols) or any(not isinstance(n, str) for n in notes):
        raise ValueError("Need single-character symbols and string auxiliary notes")
    if np.any(positions < 0) or np.any(positions >= len(values)):
        raise ValueError("Annotation position outside recording")
    positions = positions.astype(np.int64)
    if np.any(np.diff(positions) < 0):
        raise ValueError("Annotation positions must be nondecreasing")
    kept, excluded, boundary = [], [], []
    left_count = right_count = 0
    for index, (position, symbol) in enumerate(zip(positions, symbols)):
        if symbol not in BEAT_SYMBOLS:
            excluded.append(index)
            continue
        left = position < before
        right = position + after > len(values)
        if left or right:
            boundary.append(index)
            left_count += int(left)
            right_count += int(right)
        else:
            kept.append(index)
    indices = np.array(kept, dtype=np.int64)
    X = np.empty((len(kept), channels, before + after), dtype=np.float64)
    for row, index in enumerate(kept):
        position = positions[index]
        X[row] = values[position-before:position+after].T
    report = {"total_annotations": len(symbols), "heartbeat_annotations": len(kept) + len(boundary),
              "extracted_beats": len(kept), "excluded_nonheartbeat": len(excluded),
              "skipped_boundary": len(boundary), "skipped_left": left_count, "skipped_right": right_count,
              "excluded_symbol_counts": dict(Counter(symbols[i] for i in excluded)),
              "excluded_nonheartbeat_annotation_indices": excluded,
              "skipped_boundary_annotation_indices": boundary}
    config = {"before_seconds_requested": float(before_seconds), "after_seconds_requested": float(after_seconds),
              "before_samples": before, "after_samples": after, "sequence_length": before + after,
              "sampling_frequency": fs, "rounding": "nearest_half_up_independently",
              "interval": "[position-before_samples, position+after_samples)", "boundary_policy": "skip",
              "heartbeat_symbols": sorted(BEAT_SYMBOLS), "signal_processing": "input_as_supplied",
              "patient_group_source": "caller_supplied" if patient_group_id else "unknown"}
    return SegmentedHeartbeats(X, np.full(len(kept), record_id), np.full(len(kept), patient_group_id or ""),
        positions[indices].copy(), indices, np.array([symbols[i] for i in kept], dtype=str),
        np.array([notes[i] for i in kept], dtype=str), fs, names, signal_units, before, after, report, config)


def segment_recording(record: ECGRecording, *, before_seconds=0.25, after_seconds=0.45,
                      patient_group_id=None, filter_kwargs=None, normalizer=None):
    """Filter full record, normalize per record/channel, then extract expert windows.

    An optional already-fitted ECGNormalizer is transformed only (never refitted),
    allowing Anna to supply frozen train-only parameters after establishing splits.
    """
    options = {} if filter_kwargs is None else dict(filter_kwargs)
    filtered = filter_recording(record, **options)
    if normalizer is None:
        normalizer = ECGNormalizer().fit(filtered.signals, scope="recording", source_ids=[record.record_id],
                                         channel_names=record.channel_names)
    normalized = normalizer.transform(filtered.signals, channel_names=record.channel_names)
    result = segment_heartbeats(normalized, record.sampling_frequency, record.annotation_positions,
        record.annotation_symbols, record_id=record.record_id, before_seconds=before_seconds,
        after_seconds=after_seconds, channel_names=record.channel_names,
        units=("dimensionless",) * len(record.channel_names), annotation_aux_notes=record.annotation_aux_notes,
        patient_group_id=patient_group_id)
    defaults = {name: parameter.default for name, parameter in inspect.signature(filter_ecg).parameters.items()
                if parameter.default is not inspect.Parameter.empty}
    result.configuration.update({"signal_processing": "full_record_filter_then_normalize_then_segment",
                                 "filtering": {**defaults, **options}, "normalization": normalizer.to_dict()})
    return result
