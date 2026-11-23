"""Versioned configuration and transparent arithmetic for the Phase-16 baseline."""
from dataclasses import asdict,dataclass
from numbers import Integral,Real


@dataclass(frozen=True)
class RhythmCNNConfig:
    version: str = 'ptbxl-rhythm-cnn-v1'
    input_channels: int = 12
    sequence_length: int = 1000
    num_classes: int = 4
    channels: tuple[int,...] = (32,64,128)
    kernels: tuple[int,...] = (15,9,5)
    pool_size: int = 2
    head_features: int = 64
    dropout: float = 0.3

    def __post_init__(self):
        if (self.input_channels,self.sequence_length,self.num_classes)!=(12,1000,4):
            raise ValueError('This baseline is specific to PTB-XL [12,1000] and four rhythm classes')
        if len(self.channels)!=3 or len(self.kernels)!=3:
            raise ValueError('Exactly three convolution blocks are required')
        for value in (*self.channels,*self.kernels,self.head_features,self.pool_size):
            if isinstance(value,bool) or not isinstance(value,Integral) or value<1:
                raise ValueError('Channel, kernel, head and pooling sizes must be positive integers')
        if any(kernel%2==0 for kernel in self.kernels) or self.pool_size!=2:
            raise ValueError('Use odd kernels and pooling size 2')
        if isinstance(self.dropout,bool) or not isinstance(self.dropout,Real) or not 0<=self.dropout<1:
            raise ValueError('Dropout must be finite in [0,1)')

    def to_dict(self):
        return asdict(self)

    def expected_parameter_count(self):
        """Conv weights (no bias), BN scale/bias, two fully-connected layers."""
        total=0; previous=self.input_channels
        for width,kernel in zip(self.channels,self.kernels):
            total+=previous*width*kernel+2*width
            previous=width
        total+=previous*self.head_features+self.head_features
        total+=self.head_features*self.num_classes+self.num_classes
        return total
