"""Bounded LRU shard loading for MIT-BIH beats; no PTB-XL interfaces are changed."""
from collections import OrderedDict
import json
from numbers import Integral
import numpy as np
import torch
from torch.utils.data import Dataset
from .mitbih_protocol import MITBIHExportIndex,load_split,validate_split


class MITBIHBeatDataset(Dataset):
    """Lazy [2,252] float32 beats, scalar long labels, and source metadata.

    Compressed NPZ requires decompressing one recording, not one beat. A bounded
    per-process LRU avoids that cost on every access. Each worker owns its cache;
    total cache RAM therefore scales with worker count. X is copied per beat so
    in-place augmentation cannot corrupt the cached source or later samples.
    """
    def __init__(self,index=None,split=None,*,partition='train',max_cached_shards=2,
                 allow_final_test=False):
        self.index=MITBIHExportIndex() if index is None else index
        if not isinstance(self.index,MITBIHExportIndex):raise TypeError('MIT-BIH export index required')
        self.split=json.loads(json.dumps(load_split(self.index) if split is None else split))
        validate_split(self.index,self.split)
        if partition not in ('train','validation','test'):raise ValueError('Unknown partition')
        if not isinstance(allow_final_test,bool):raise ValueError('Final-test opt-in must be an explicit boolean')
        if partition=='test' and not allow_final_test:raise PermissionError('Final MIT-BIH test is locked')
        if isinstance(max_cached_shards,bool) or not isinstance(max_cached_shards,int) or max_cached_shards<1:
            raise ValueError('Positive bounded shard-cache size required')
        self.partition=partition;self.max_cached_shards=max_cached_shards
        self.allow_final_test=bool(allow_final_test);self.cache=OrderedDict()
        selected=set(self.split['groups'][partition])
        self.shards=tuple(e for e in self.index.entries if e.patient_group in selected)
        self.stops=np.cumsum([e.count for e in self.shards],dtype=np.int64)

    def __len__(self):return int(self.stops[-1])

    def source_location(self,position):
        if isinstance(position,bool) or not isinstance(position,Integral) or not 0<=position<len(self):
            raise IndexError('Beat index must be an integer within this partition')
        j=int(np.searchsorted(self.stops,int(position),side='right'))
        start=0 if j==0 else int(self.stops[j-1])
        return self.shards[j],int(position)-start

    def _arrays(self,entry):
        path=self.index.root/entry.file;stat=path.stat();stamp=(stat.st_size,stat.st_mtime_ns)
        if entry.record_id in self.cache:
            arrays,original_stamp=self.cache.pop(entry.record_id)
            if stamp!=original_stamp:raise ValueError(f'Cached shard changed on disk: {entry.record_id}')
        else:
            arrays=self.index.load_shard(entry.record_id,self.split,self.partition,
                                          allow_final_test=self.allow_final_test)
            after=path.stat()
            if stamp!=(after.st_size,after.st_mtime_ns):raise ValueError('Shard changed while loading')
        self.cache[entry.record_id]=(arrays,stamp)
        while len(self.cache)>self.max_cached_shards:self.cache.popitem(last=False)
        return arrays

    def __getitem__(self,position):
        entry,row=self.source_location(position);arrays=self._arrays(entry)
        x=torch.from_numpy(arrays['X'][row].copy())
        y=torch.tensor(int(arrays['y'][row]),dtype=torch.long)
        return {'X':x,'y':y,'record_id':entry.record_id,'patient_group':entry.patient_group,
                'original_symbol':str(arrays['original_symbols'][row]),
                'annotation_position':int(arrays['annotation_positions'][row]),
                'annotation_index':int(arrays['annotation_indices'][row]),
                'channel_names':tuple(arrays['channel_names'].tolist()),
                'sampling_frequency':float(arrays['sampling_frequency']),
                'source_row':row,'global_source_index':entry.global_start+row}


def collate_mitbih(samples):
    """Keep string/lead metadata aligned without the default tuple transposition."""
    if not samples:raise ValueError('Nonempty batch required')
    result={'X':torch.stack([s['X'] for s in samples]),'y':torch.stack([s['y'] for s in samples])}
    for key in ('record_id','patient_group','original_symbol','channel_names'):
        result[key]=[s[key] for s in samples]
    for key in ('annotation_position','annotation_index','source_row','global_source_index'):
        result[key]=torch.tensor([s[key] for s in samples],dtype=torch.long)
    result['sampling_frequency']=torch.tensor([s['sampling_frequency'] for s in samples],dtype=torch.float32)
    return result
