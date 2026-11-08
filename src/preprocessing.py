"""Offline ECG filtering and normalization; samples on axis 0, no input mutation."""
from dataclasses import replace
from numbers import Integral, Real
import numpy as np
from scipy import signal
from .data_loader import ECGRecording


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a positive finite number")
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a positive finite number") from None
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return value


def _input(values, fs):
    fs = _positive(fs, "sampling_frequency")
    raw = np.asarray(values)
    if raw.dtype.kind not in "iuf" or raw.ndim not in (1, 2) or raw.size == 0:
        raise ValueError("Signal must be a nonempty real numeric [samples] or [samples, channels] array")
    values = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Signal contains non-finite values")
    return values, fs


def _frequency(value, fs, name):
    value = _positive(value, name)
    if value >= fs / 2:
        raise ValueError(f"{name} must be below Nyquist ({fs / 2} Hz)")
    return value


def _apply(values, sos):
    # Match scipy.signal.sosfiltfilt's documented default padding requirement.
    padlen = 3 * (2 * len(sos) + 1 - min((sos[:, 2] == 0).sum(), (sos[:, 5] == 0).sum()))
    if len(values) <= padlen:
        raise ValueError(f"Signal too short for zero-phase filtering: need > {padlen} samples, got {len(values)}")
    result = signal.sosfiltfilt(sos, values, axis=0, padlen=int(padlen))
    if not np.isfinite(result).all():
        raise ValueError("Filtering produced non-finite values")
    return result


def _butterworth(values, sampling_frequency, cutoff_hz, order, kind):
    values, fs = _input(values, sampling_frequency)
    cutoff = _frequency(cutoff_hz, fs, "cutoff_hz")
    if isinstance(order, (bool, np.bool_)) or not isinstance(order, Integral) or order < 1:
        raise ValueError("order must be a positive integer")
    sos = signal.butter(int(order), cutoff, btype=kind, fs=fs, output="sos")
    return _apply(values, sos)


def highpass_filter(values, sampling_frequency, cutoff_hz=0.5, order=4):
    """Remove slow baseline variation using a Butterworth SOS filter."""
    return _butterworth(values, sampling_frequency, cutoff_hz, order, "highpass")


def lowpass_filter(values, sampling_frequency, cutoff_hz=45.0, order=4):
    """Reduce high-frequency noise using a Butterworth SOS filter."""
    return _butterworth(values, sampling_frequency, cutoff_hz, order, "lowpass")


def notch_filter(values, sampling_frequency, frequency_hz=50.0, quality_factor=30.0):
    """Explicit optional powerline rejection; frequency must be 50 or 60 Hz."""
    values, fs = _input(values, sampling_frequency)
    frequency = _frequency(frequency_hz, fs, "frequency_hz")
    if frequency not in (50.0, 60.0):
        raise ValueError("Powerline frequency must be 50 or 60 Hz")
    quality = _positive(quality_factor, "quality_factor")
    b, a = signal.iirnotch(frequency, quality, fs=fs)
    return _apply(values, signal.tf2sos(b, a))


def filter_ecg(values, sampling_frequency, *, highpass_hz=0.5, lowpass_hz=45.0,
               order=4, notch_hz=None, quality_factor=30.0):
    """Filter offline, optionally notch FIRST, then high-pass and low-pass.

    Set a cutoff to None to disable that stage. Notch is disabled by default:
    inspect raw and post-bandpass powerline spectra before explicitly enabling it.
    Forward/backward filtering is noncausal and doubles magnitude-response order.
    Returns float64 with unchanged shape and sample timing; no resampling occurs.
    """
    values, fs = _input(values, sampling_frequency)
    if highpass_hz is not None:
        _frequency(highpass_hz, fs, "highpass_hz")
    if lowpass_hz is not None:
        _frequency(lowpass_hz, fs, "lowpass_hz")
    if highpass_hz is not None and lowpass_hz is not None and highpass_hz >= lowpass_hz:
        raise ValueError("highpass_hz must be lower than lowpass_hz")
    result = values.copy()
    if notch_hz is not None:
        result = notch_filter(result, fs, notch_hz, quality_factor)
    if highpass_hz is not None:
        result = highpass_filter(result, fs, highpass_hz, order)
    if lowpass_hz is not None:
        result = lowpass_filter(result, fs, lowpass_hz, order)
    return result


def filter_recording(record: ECGRecording, **kwargs) -> ECGRecording:
    """Return a recording with filtered samples and unchanged metadata."""
    return replace(record, signals=filter_ecg(record.signals, record.sampling_frequency, **kwargs))


def powerline_diagnostics(values, sampling_frequency, frequency_hz=50.0):
    """Return per-channel Welch peak/background ratios near a powerline frequency.

    Heuristic evidence only, not an automatic notch decision. Use both raw and
    bandpass-only signals; biological activity may also create spectral peaks.
    """
    values, fs = _input(values, sampling_frequency)
    frequency = _frequency(frequency_hz, fs, "frequency_hz")
    if frequency not in (50.0, 60.0) or frequency + 5 >= fs / 2:
        raise ValueError("Need 50/60 Hz and at least 5 Hz of space below Nyquist")
    if len(values) < int(np.ceil(4 * fs)):
        raise ValueError("Powerline diagnosis needs at least four seconds")
    frequencies, psd = signal.welch(values, fs=fs, nperseg=int(round(4 * fs)), axis=0)
    if psd.ndim == 1:
        psd = psd[:, None]
    offset = np.abs(frequencies - frequency)
    peak = psd[offset <= 0.5].max(axis=0)
    background = np.median(psd[(offset >= 2) & (offset <= 5)], axis=0)
    ratio = peak / np.maximum(background, np.finfo(float).tiny)
    return {"frequency_hz": frequency, "peak_psd": peak.tolist(),
            "background_psd": background.tolist(), "peak_to_background_ratio": ratio.tolist()}


class ECGNormalizer:
    """Independent channel normalization with reusable, JSON-serializable state.

    fit() requires an explicit statistics scope. Training provenance is supplied
    by the caller: this class cannot verify Anna's patient split. Never fit on
    validation/test patients for the training-statistics workflow.
    """

    def __init__(self, method="zscore", *, epsilon=1e-8, feature_range=(0.0, 1.0), clip=False):
        if method not in ("zscore", "minmax"):
            raise ValueError("method must be 'zscore' or 'minmax'")
        self.method = method
        self.epsilon = _positive(epsilon, "epsilon")
        bounds = np.asarray(feature_range)
        if bounds.shape != (2,) or bounds.dtype.kind not in "iuf":
            raise ValueError("feature_range must contain two finite increasing numbers")
        bounds = bounds.astype(float)
        with np.errstate(over="ignore"):
            width = bounds[1] - bounds[0]
        if not np.isfinite(bounds).all() or not np.isfinite(width) or width <= 0:
            raise ValueError("feature_range must contain two finite increasing numbers")
        if not isinstance(clip, bool):
            raise ValueError("clip must be bool")
        if clip and method != "minmax":
            raise ValueError("clip is available only for minmax")
        self.feature_range = tuple(bounds.tolist())
        self.clip = clip
        self._state = None

    @staticmethod
    def _values(values):
        # Reuse the filtering input checks; normalization does not resample.
        return _input(values, 1.0)[0]

    @staticmethod
    def _names(channel_names, channels):
        if channel_names is None:
            return None
        if isinstance(channel_names, str):
            raise ValueError("channel_names must be an ordered list of channel names")
        names = list(channel_names)
        if len(names) != channels or any(not isinstance(n, str) or not n for n in names):
            raise ValueError("channel_names must match the channel count")
        return names

    def fit(self, values, *, scope, source_ids, channel_names=None):
        """Fit along axis 0 on ONE recording/segment or caller-provided training data.

        scope: 'recording', 'segment', or 'training'. source_ids records provenance,
        not patient identity. Supplying training scope is the caller's declaration
        that all supplied samples belong exclusively to the established train set.
        """
        if scope not in ("recording", "segment", "training"):
            raise ValueError("scope must be recording, segment, or training")
        if isinstance(source_ids, str):
            raise ValueError("source_ids must be a nonempty sequence of identifiers")
        ids = list(source_ids)
        if not ids or any(not isinstance(rid, str) or not rid for rid in ids):
            raise ValueError("source_ids must contain nonempty string identifiers")
        values = self._values(values)
        channels = 1 if values.ndim == 1 else values.shape[1]
        names = self._names(channel_names, channels)
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                if self.method == "zscore":
                    offset = np.atleast_1d(np.mean(values, axis=0))
                    scale = np.atleast_1d(np.std(values, axis=0, ddof=0))
                else:
                    offset = np.atleast_1d(np.min(values, axis=0))
                    scale = np.atleast_1d(np.max(values, axis=0)) - offset
        except FloatingPointError as exc:
            raise ValueError("Normalization statistics exceed numeric range") from exc
        if not np.isfinite(offset).all() or not np.isfinite(scale).all():
            raise ValueError("Normalization statistics must be finite")
        self._state = {"scope": scope, "source_ids": ids, "channel_names": names,
                       "sample_count": len(values), "offset": offset.tolist(), "scale": scale.tolist()}
        return self

    def transform(self, values, *, channel_names=None):
        """Apply stored parameters without refitting or mutating input/state.

        Near-constant fitted channels (scale <= epsilon) become 0 for Z-score,
        or the lower feature-range bound for Min-Max, including at inference.
        """
        if self._state is None:
            raise ValueError("Normalizer must be fitted or loaded before transform")
        values = self._values(values)
        channels = 1 if values.ndim == 1 else values.shape[1]
        if channels != len(self._state["offset"]):
            raise ValueError("Channel count does not match fitted parameters")
        names = self._names(channel_names, channels)
        if names != self._state["channel_names"]:
            raise ValueError("Channel names/order must match fitted parameters")
        offset = np.array(self._state["offset"])
        scale = np.array(self._state["scale"])
        inactive = scale <= self.epsilon
        divisor = np.where(inactive, 1.0, scale)
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                result = (values - offset) / divisor
                if values.ndim == 1:
                    result = result.reshape(values.shape)
                    if inactive[0]:
                        result.fill(0)
                else:
                    result[:, inactive] = 0
                if self.method == "minmax":
                    low, high = self.feature_range
                    result = result * (high - low) + low
                    if self.clip:
                        result = np.clip(result, low, high)
        except FloatingPointError as exc:
            raise ValueError("Normalized samples exceed numeric range") from exc
        if not np.isfinite(result).all():
            raise ValueError("Normalized samples must be finite")
        return result

    def to_dict(self):
        """Export a detached JSON-compatible state including provenance."""
        import copy
        if self._state is None:
            raise ValueError("Normalizer is not fitted")
        return copy.deepcopy({"version": 1, "method": self.method, "epsilon": self.epsilon,
                              "feature_range": list(self.feature_range), "clip": self.clip,
                              **self._state})

    @classmethod
    def from_dict(cls, state):
        """Load validated parameters; no data are used to recompute statistics."""
        import copy
        required = {"version", "method", "epsilon", "feature_range", "clip", "scope", "source_ids",
                    "channel_names", "sample_count", "offset", "scale"}
        if not isinstance(state, dict) or set(state) != required or state["version"] != 1:
            raise ValueError("Invalid normalization state schema/version")
        obj = cls(state["method"], epsilon=state["epsilon"], feature_range=state["feature_range"], clip=state["clip"])
        offset, scale = np.asarray(state["offset"]), np.asarray(state["scale"])
        if (offset.ndim != 1 or not offset.size or scale.shape != offset.shape
                or offset.dtype.kind not in "iuf" or scale.dtype.kind not in "iuf"):
            raise ValueError("Invalid normalization parameter arrays")
        if not np.isfinite(offset).all() or not np.isfinite(scale).all() or np.any(scale < 0):
            raise ValueError("Invalid normalization parameter values")
        if state["scope"] not in ("recording", "segment", "training"):
            raise ValueError("Invalid statistics scope")
        ids = state["source_ids"]
        if not isinstance(ids, list) or not ids or any(not isinstance(s, str) or not s for s in ids):
            raise ValueError("Invalid source_ids")
        count = state["sample_count"]
        if isinstance(count, bool) or not isinstance(count, Integral) or count < 1:
            raise ValueError("Invalid sample_count")
        obj._names(state["channel_names"], len(offset))
        obj._state = copy.deepcopy({k: state[k] for k in required - {"version", "method", "epsilon", "feature_range", "clip"}})
        return obj

    def save(self, path):
        """Write JSON without overwriting an existing file."""
        import json
        from pathlib import Path
        state = self.to_dict()
        with Path(path).open("x") as output:
            json.dump(state, output, indent=2, allow_nan=False)

    @classmethod
    def load(cls, path):
        """Restore parameters for later inference."""
        import json
        from pathlib import Path
        with Path(path).open() as source:
            return cls.from_dict(json.load(source))


def normalize_ecg(values, method="zscore", *, scope="recording", source_ids=("unspecified",),
                  channel_names=None, epsilon=1e-8, feature_range=(0.0, 1.0), clip=False):
    """Fit/transform only this array, per channel; default scope is one recording.

    Never pass combined validation/test patients here to compute global statistics.
    For training-statistics inference use ECGNormalizer.fit then transform instead.
    Returns an array; use ECGNormalizer directly to retain fitted parameters.
    """
    normalizer = ECGNormalizer(method, epsilon=epsilon, feature_range=feature_range, clip=clip)
    normalizer.fit(values, scope=scope, source_ids=source_ids, channel_names=channel_names)
    return normalizer.transform(values, channel_names=channel_names)


def zscore_normalize(values, **kwargs):
    """Per-channel Z-score, using population standard deviation (ddof=0)."""
    return normalize_ecg(values, method="zscore", **kwargs)


def minmax_normalize(values, **kwargs):
    """Per-channel Min-Max scaling; default range [0, 1]."""
    return normalize_ecg(values, method="minmax", **kwargs)
