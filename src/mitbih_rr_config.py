"""Versioned causal RR contract. This does not make ECG filtering causal."""
from dataclasses import dataclass,asdict

RAW_RR_NAMES=('previous_rr_seconds','preceding_baseline_rr_seconds','previous_over_baseline')
RR_FEATURE_NAMES=RAW_RR_NAMES+('previous_rr_available','baseline_rr_available','ratio_available')

@dataclass(frozen=True)
class RRFeatureConfig:
    version: str='mitbih-causal-rr-v1'
    history_intervals: int=8
    epsilon: float=1e-8

    def __post_init__(self):
        if self.version!='mitbih-causal-rr-v1' or self.history_intervals!=8 or self.epsilon!=1e-8:
            raise ValueError('Use the frozen v1 RR configuration; changes require a new version')
    def to_dict(self):return asdict(self)
