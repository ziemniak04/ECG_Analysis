"""Small randomly initialized 1D-CNN baseline. No training loop or pretrained weights."""
import torch
from torch import nn
from ..ptbxl_model_config import RhythmCNNConfig


class PTBXLRhythmCNN(nn.Module):
    """Input float32 [B,12,1000], output four unnormalized class logits."""
    def __init__(self,config=None):
        super().__init__()
        self.config=RhythmCNNConfig() if config is None else config
        if not isinstance(self.config,RhythmCNNConfig):raise TypeError('Expected RhythmCNNConfig')
        blocks=[];previous=self.config.input_channels
        for width,kernel in zip(self.config.channels,self.config.kernels):
            blocks.append(nn.Sequential(
                nn.Conv1d(previous,width,kernel,padding=kernel//2,bias=False),
                nn.BatchNorm1d(width),nn.ReLU(),nn.MaxPool1d(self.config.pool_size)))
            previous=width
        self.features=nn.Sequential(*blocks)
        self.pool=nn.AdaptiveAvgPool1d(1)
        self.head=nn.Sequential(nn.Flatten(),nn.Linear(previous,self.config.head_features),
            nn.ReLU(),nn.Dropout(self.config.dropout),nn.Linear(self.config.head_features,self.config.num_classes))
        if self.parameter_count()!=self.config.expected_parameter_count():
            raise RuntimeError('Model parameter count disagrees with versioned configuration')

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self,x):
        if x.ndim!=3 or tuple(x.shape[1:])!=(12,1000) or x.shape[0]<1:
            raise ValueError('Expected nonempty ECG batch [B,12,1000]')
        if x.dtype!=torch.float32:raise TypeError('Expected float32 ECG input')
        if not torch.isfinite(x).all():raise ValueError('ECG batch contains nonfinite values')
        return self.head(self.pool(self.features(x)))
