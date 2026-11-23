"""PyTorch-dependent tests. Missing torch is an explicit skip, never a fake pass."""
import json
from pathlib import Path
import pytest
import numpy as np

torch=pytest.importorskip('torch',reason='PyTorch CPU installation is required for Phase-16 tensor/model verification')
from torch.utils.data import DataLoader
from src.ptbxl_protocol import PTBXLExportIndex
from src.ptbxl_torch import PTBXLRhythmDataset
from src.models.ptbxl_cnn import PTBXLRhythmCNN
from src.ptbxl_model_config import RhythmCNNConfig


@pytest.fixture(scope='module')
def real_index():
    torch.set_num_threads(1)
    return PTBXLExportIndex()


def test_lazy_real_dataset_and_metadata(real_index,monkeypatch):
    calls=[];original=real_index.load_numpy
    def observed(*args,**kwargs):
        calls.append(args[0]);return original(*args,**kwargs)
    monkeypatch.setattr(real_index,'load_numpy',observed)
    dataset=PTBXLRhythmDataset(real_index,partition='train')
    assert calls==[]
    item=dataset[0]
    assert len(calls)==1
    assert item['X'].shape==(12,1000) and item['X'].dtype==torch.float32
    assert item['y'].ndim==0 and item['y'].dtype==torch.long
    assert 0<=int(item['y'])<=3 and 1<=item['strat_fold']<=8
    assert item['ecg_id']==dataset.metadata(0).ecg_id
    assert item['patient_id']==dataset.metadata(0).patient_id


@pytest.mark.parametrize('batch_size',[1,4])
def test_model_real_training_batch_output(real_index,batch_size):
    batch=next(iter(DataLoader(PTBXLRhythmDataset(real_index,partition='train'),batch_size=batch_size,shuffle=False)))
    model=PTBXLRhythmCNN();model.eval()
    with torch.no_grad():logits=model(batch['X'])
    assert logits.shape==(batch_size,4) and torch.isfinite(logits).all()
    assert model.parameter_count()==74116==RhythmCNNConfig().expected_parameter_count()
    assert not any(isinstance(layer,torch.nn.Softmax) for layer in model.modules())


def test_real_training_batch_backward_gradients(real_index):
    batch=next(iter(DataLoader(PTBXLRhythmDataset(real_index,partition='train'),batch_size=4,shuffle=False)))
    model=PTBXLRhythmCNN();model.train()
    loss=torch.nn.CrossEntropyLoss()(model(batch['X']),batch['y'])
    assert torch.isfinite(loss)
    loss.backward()
    gradients=[p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in gradients)
    assert sum(float(g.norm()) for g in gradients)>0
    assert all(1<=int(fold)<=8 for fold in batch['strat_fold'])
    # No optimizer.step here: the sole update belongs to phase16_model_smoke.


def test_deterministic_eval_and_input_validation(real_index):
    x=PTBXLRhythmDataset(real_index,partition='train')[0]['X'].unsqueeze(0)
    torch.manual_seed(42);model=PTBXLRhythmCNN();model.eval()
    with torch.no_grad():a,b=model(x),model(x)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    with pytest.raises(ValueError):model(x[:,:,:999])
    with pytest.raises(TypeError):model(x.double())
    invalid=x.clone();invalid[0,0,0]=float('nan')
    with pytest.raises(ValueError):model(invalid)
