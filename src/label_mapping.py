"""Explicit AAMI-style five-class labels over existing segmented heartbeats.

Reference mapping: Das & Ari (2014), Table 1, doi:10.1155/2014/178436.
WFDB symbols: https://archive.physionet.org/physiobank/annotations.shtml
This is label preparation, not a claim of EC57 evaluation-protocol compliance.
"""
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
import numpy as np
from .segmentation import SegmentedHeartbeats, segment_recording
from .data_loader import DEFAULT_DATA_DIR, list_recordings, load_recording
from .visualization import BEAT_SYMBOLS

CLASS_NAMES = ('N', 'S', 'V', 'F', 'Q')
CLASS_INDICES = MappingProxyType({name: i for i, name in enumerate(CLASS_NAMES)})
DEFAULT_MAPPING = MappingProxyType({symbol: name for name, symbols in
    [('N', ('N','L','R','e','j')), ('S', ('A','a','J','S')), ('V', ('V','E')),
     ('F', ('F',)), ('Q', ('/','f','Q'))] for symbol in symbols})
UNMAPPED_INDEX = -1


def validate_mapping(mapping=None):
    """Return a deterministic detached map. Custom maps replace, not extend, defaults.

    B, n and r are real beat symbols but intentionally unmapped by default.
    Non-beat symbols cannot be added as classes through this interface.
    """
    mapping = DEFAULT_MAPPING if mapping is None else mapping
    if not isinstance(mapping, Mapping):
        raise ValueError('mapping must be a symbol-to-class mapping')
    for symbol, name in mapping.items():
        if not isinstance(symbol, str) or symbol not in BEAT_SYMBOLS:
            raise ValueError(f'Not an accepted segmented heartbeat symbol: {symbol!r}')
        if not isinstance(name, str) or name not in CLASS_NAMES:
            raise ValueError(f'Invalid AAMI class for {symbol!r}: {name!r}')
    return dict(sorted(mapping.items()))


def map_symbol(symbol, mapping=None):
    """Map one original symbol strictly; never default unknown symbols to Normal/Q."""
    selected = validate_mapping(mapping)
    if not isinstance(symbol, str) or symbol not in selected:
        raise ValueError(f'Unmapped annotation symbol: {symbol!r}')
    name = selected[symbol]
    return name, CLASS_INDICES[name]


@dataclass(frozen=True)
class LabeledHeartbeats:
    """One-to-one labels alongside untouched segments and all their metadata.

    Under policy 'mark', -1/UNMAPPED is a sentinel, NOT a sixth training class.
    Always check mapped_mask before any future model use.
    """
    segments: SegmentedHeartbeats
    y: np.ndarray
    mapped_classes: np.ndarray
    class_names: tuple[str, ...]
    mapping: dict
    unknown_policy: str

    @property
    def X(self):
        return self.segments.X

    @property
    def mapped_mask(self):
        return self.y >= 0


def map_segments(segments, mapping=None, *, unknown_policy='error'):
    """Preserve every segment, its order and metadata; attach deterministic labels.

    'error' (default): stop on any unmapped symbol.
    'mark': retain it with y=-1 and mapped_classes='UNMAPPED'; report explicitly.
    Neither policy silently drops windows or invents labels/patient identities.
    """
    if unknown_policy not in ('error','mark'):
        raise ValueError("unknown_policy must be 'error' or 'mark'")
    selected = validate_mapping(mapping)
    if not isinstance(segments, SegmentedHeartbeats) or segments.X.ndim != 3:
        raise ValueError('Expected SegmentedHeartbeats with 3D X')
    count = len(segments.X)
    fields = (segments.record_ids, segments.patient_group_ids, segments.annotation_positions,
              segments.annotation_indices, segments.annotation_symbols, segments.annotation_aux_notes)
    if any(np.asarray(field).ndim != 1 or len(field) != count for field in fields):
        raise ValueError('Segment metadata lengths must match X')
    if len(segments.channel_names) != segments.X.shape[1] or len(segments.units) != segments.X.shape[1]:
        raise ValueError('Channel metadata must match X')
    unmapped = Counter(s for s in segments.annotation_symbols if s not in selected)
    if unmapped and unknown_policy == 'error':
        raise ValueError(f'Unmapped segmented symbols (no fallback): {dict(sorted(unmapped.items()))}')
    y = np.array([CLASS_INDICES[selected[s]] if s in selected else UNMAPPED_INDEX
                  for s in segments.annotation_symbols], dtype=np.int64)
    classes = np.array([CLASS_NAMES[i] if i >= 0 else 'UNMAPPED' for i in y], dtype='<U8')
    return LabeledHeartbeats(segments, y, classes, CLASS_NAMES, selected, unknown_policy)


def _count_summary(class_counts, symbol_counts, unmapped_counts):
    mapped = sum(class_counts.values())
    unmapped = sum(unmapped_counts.values())
    total = mapped + unmapped
    return {'segment_count': total, 'mapped_count': mapped, 'unmapped_count': unmapped,
            'class_counts': {name: int(class_counts.get(name, 0)) for name in CLASS_NAMES},
            'class_percentages_of_mapped': {name: 100*class_counts.get(name, 0)/mapped if mapped else 0.0
                                           for name in CLASS_NAMES},
            'class_percentages_of_segments': {name: 100*class_counts.get(name, 0)/total if total else 0.0
                                             for name in CLASS_NAMES},
            'original_segment_symbol_counts': dict(sorted(symbol_counts.items())),
            'unmapped_symbol_counts': dict(sorted(unmapped_counts.items()))}


def class_distribution(labeled):
    """Summarize actual segment labels; include absent classes and explicit denominators."""
    symbols = labeled.segments.annotation_symbols
    counts = Counter(labeled.mapped_classes[labeled.mapped_mask].tolist())
    unmapped = Counter(symbols[~labeled.mapped_mask].tolist())
    result = _count_summary(counts, Counter(symbols.tolist()), unmapped)
    if result['segment_count'] != len(labeled.X):
        raise ValueError('Counts do not match segment count')
    result['patient_groups_unresolved_count'] = int(np.count_nonzero(labeled.segments.patient_group_ids == ''))
    result['segmentation_report'] = labeled.segments.report
    return result


def summarize_recordings(root=DEFAULT_DATA_DIR, record_ids=None, *, mapping=None):
    """Summarize all local complete records by default, processing one at a time.

    Reuses the actual default preprocessing/segmentation, so counts include its
    boundary exclusions. No dataset concatenation, balancing or splitting occurs.
    Returns small counts/metadata, not ECG arrays. Unmapped symbols are marked.
    """
    selected = validate_mapping(mapping)
    ids = list_recordings(root) if record_ids is None else list(record_ids)
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate recording IDs would double-count beats')
    records = {}
    counts, symbols, unmapped = Counter(), Counter(), Counter()
    for rid in ids:
        record = load_recording(rid, root)
        segments = segment_recording(record)
        summary = class_distribution(map_segments(segments, selected, unknown_policy='mark'))
        summary['original_all_annotation_symbol_counts'] = dict(sorted(Counter(record.annotation_symbols).items()))
        records[rid] = summary
        counts.update(summary['class_counts'])
        symbols.update(summary['original_segment_symbol_counts'])
        unmapped.update(summary['unmapped_symbol_counts'])
    return {'class_names': list(CLASS_NAMES), 'class_indices': dict(CLASS_INDICES), 'mapping': selected,
            'unknown_policy': 'mark', 'record_ids': ids, 'records': records,
            'aggregate': _count_summary(counts, symbols, unmapped)}
