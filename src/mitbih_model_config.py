"""Independent five-class heartbeat baseline contract and parameter arithmetic."""
from dataclasses import dataclass,asdict
from numbers import Integral,Real


@dataclass(frozen=True)
class BeatCNNConfig:
    version: str='mitbih-beat-cnn-v1'
    input_channels: int=2
    sequence_length: int=252
    num_classes: int=5
    channels: tuple[int,...]=(16,32,64)
    kernels: tuple[int,...]=(15,9,5)
    pool_size: int=2
    head_features: int=32
    dropout: float=.3

    def __post_init__(self):
        if (self.input_channels,self.sequence_length,self.num_classes)!=(2,252,5):
            raise ValueError('Heartbeat baseline expects [2,252] and five classes')
        if len(self.channels)!=3 or len(self.kernels)!=3:raise ValueError('Three convolution blocks required')
        if any(isinstance(v,bool) or not isinstance(v,Integral) or v<1
               for v in (*self.channels,*self.kernels,self.pool_size,self.head_features)):
            raise ValueError('Positive integer layer sizes required')
        if self.pool_size!=2 or any(k%2==0 for k in self.kernels):raise ValueError('Pool=2 and odd kernels required')
        if isinstance(self.dropout,bool) or not isinstance(self.dropout,Real) or not 0<=self.dropout<1:
            raise ValueError('Dropout must be in [0,1)')

    def to_dict(self):return asdict(self)

    def expected_parameter_count(self):
        previous=self.input_channels;total=0
        for width,kernel in zip(self.channels,self.kernels):
            total+=previous*width*kernel+2*width;previous=width
        return total+previous*self.head_features+self.head_features+self.head_features*self.num_classes+self.num_classes
