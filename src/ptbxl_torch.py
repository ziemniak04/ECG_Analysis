"""Lazy PyTorch Dataset for immutable, previously preprocessed PTB-XL NPZ records."""
from numbers import Integral
import torch
from torch.utils.data import Dataset
from .ptbxl_protocol import PTBXLExportIndex,PARTITION_FOLDS


class PTBXLRhythmDataset(Dataset):
    """Return X/y tensors plus original integer IDs/fold, one record per access.

    No waveform cache: bounded per-record RAM. Metadata stays in the supplied
    index. The default checksum verification catches changed exports on access.
    Partition is an official fold group or None for the whole metadata index.
    """
    def __init__(self,index=None,*,partition=None,verify_checksum=True):
        self.index=PTBXLExportIndex() if index is None else index
        if not isinstance(self.index,PTBXLExportIndex):raise TypeError('Expected PTBXLExportIndex')
        if partition is not None and partition not in PARTITION_FOLDS:
            raise ValueError('Partition must be train, validation, test or None')
        self.partition=partition
        self.source_indices=tuple(range(len(self.index))) if partition is None else getattr(self.index.partitions,partition)
        self.verify_checksum=bool(verify_checksum)

    def __len__(self):return len(self.source_indices)

    def metadata(self,position):
        if isinstance(position,bool) or not isinstance(position,Integral) or not 0<=position<len(self):
            raise IndexError('Dataset index must be an integer in range')
        return self.index.entries[self.source_indices[int(position)]]

    def __getitem__(self,position):
        entry=self.metadata(position)
        arrays=self.index.load_numpy(self.source_indices[position],verify_checksum=self.verify_checksum)
        # Existing reader already guarantees float32/int64, shape and finite data.
        # from_numpy shares only this newly loaded per-record array, never stored files.
        x=torch.from_numpy(arrays['X'])
        y=torch.from_numpy(arrays['y'])
        if x.dtype!=torch.float32 or tuple(x.shape)!=(12,1000) or y.dtype!=torch.long or y.ndim!=0:
            raise ValueError(f'Invalid tensor contract for ecg_id={entry.ecg_id}')
        return {'X':x,'y':y,'ecg_id':entry.ecg_id,'patient_id':entry.patient_id,'strat_fold':entry.strat_fold}
