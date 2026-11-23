"""Separate 64-morphology + 6-RR fusion model. No pretrained initialization."""
import torch
from torch import nn
from .mitbih_cnn import MITBIHBeatCNN
from ..mitbih_rr_config import RR_FEATURE_NAMES
from ..mitbih_model_config import BeatCNNConfig


class MITBIHBeatCNNRR(nn.Module):
    def __init__(self):
        super().__init__()
        baseline=MITBIHBeatCNN()  # Same morphology initialization path, new head below.
        self.features=baseline.features;self.pool=baseline.pool
        self.head=nn.Sequential(nn.Linear(64+len(RR_FEATURE_NAMES),32),nn.ReLU(),nn.Dropout(.3),nn.Linear(32,5))
        if self.parameter_count()!=17989:raise RuntimeError('CNN+RR parameter count differs')

    def parameter_count(self):return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def configuration(self):
        return dict(version='mitbih-cnn-rr-v1',morphology=BeatCNNConfig().to_dict(),
                    rr_feature_names=list(RR_FEATURE_NAMES),fusion_width=70,head_hidden=32,dropout=.3,
                    parameter_count=self.parameter_count(),baseline_parameter_count=17797)

    def forward(self,x,rr):
        if x.ndim!=3 or tuple(x.shape[1:])!=(2,252) or not len(x):raise ValueError('ECG [B,2,252] required')
        if rr.shape!=(len(x),6):raise ValueError('RR [B,6] required')
        if x.dtype!=torch.float32 or rr.dtype!=torch.float32:raise TypeError('float32 inputs required')
        if x.device!=rr.device:raise ValueError('Inputs must share one device')
        if not torch.isfinite(x).all() or not torch.isfinite(rr).all():raise ValueError('Nonfinite inputs')
        if not bool(((rr[:,3:]==0)|(rr[:,3:]==1)).all()):raise ValueError('RR mask columns must be binary')
        morphology=self.pool(self.features(x)).flatten(1)
        return self.head(torch.cat((morphology,rr),dim=1))
