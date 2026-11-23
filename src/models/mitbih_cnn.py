"""Separate, compact heartbeat 1D-CNN. No pretrained weights or training loop."""
import torch
from torch import nn
from ..mitbih_model_config import BeatCNNConfig


class MITBIHBeatCNN(nn.Module):
    """float32 [B,2,252] -> [B,5] logits in N,S,V,F,Q order."""
    def __init__(self,config=None):
        super().__init__();self.config=BeatCNNConfig() if config is None else config
        if not isinstance(self.config,BeatCNNConfig):raise TypeError('BeatCNNConfig required')
        blocks=[];previous=self.config.input_channels
        for width,kernel in zip(self.config.channels,self.config.kernels):
            blocks.append(nn.Sequential(nn.Conv1d(previous,width,kernel,padding=kernel//2,bias=False),
                nn.BatchNorm1d(width),nn.ReLU(),nn.MaxPool1d(2)))
            previous=width
        self.features=nn.Sequential(*blocks);self.pool=nn.AdaptiveAvgPool1d(1)
        self.head=nn.Sequential(nn.Flatten(),nn.Linear(previous,self.config.head_features),nn.ReLU(),
            nn.Dropout(self.config.dropout),nn.Linear(self.config.head_features,5))
        if self.parameter_count()!=self.config.expected_parameter_count():raise RuntimeError('Parameter contract differs')

    def parameter_count(self):return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self,x):
        if x.ndim!=3 or tuple(x.shape[1:])!=(2,252) or not len(x):raise ValueError('Expected [B,2,252]')
        if x.dtype!=torch.float32:raise TypeError('float32 ECG required')
        if not torch.isfinite(x).all():raise ValueError('Nonfinite ECG input')
        return self.head(self.pool(self.features(x)))
