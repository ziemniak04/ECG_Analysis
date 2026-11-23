"""Phase-22 reporting/preflight checks; fixtures do not evaluate real test data."""
from dataclasses import replace
import json
import numpy as np
import pytest
from src.mitbih_full_run import verify_config, validated_history, preflight
from src.mitbih_training import MITBIHTrainingConfig
from src.mitbih_metrics import classification_metrics
from src.prepare_dataset import sha256_file


def test_exact_approved_config():
    verify_config(MITBIHTrainingConfig())
    for kwargs in (dict(loss_strategy='balanced'),dict(batch_size=64),dict(seed=43),dict(epochs=14)):
        with pytest.raises(ValueError,match='approved'):verify_config(replace(MITBIHTrainingConfig(),**kwargs))


def test_preflight_refuses_existing_output_before_dataset_io(tmp_path,monkeypatch):
    import src.mitbih_full_run as module
    def forbidden():raise AssertionError('Dataset must not be opened')
    monkeypatch.setattr(module,'MITBIHExportIndex',forbidden)
    output=tmp_path/'existing';output.mkdir()
    with pytest.raises(FileExistsError):preflight(MITBIHTrainingConfig(),tmp_path/'audit',output)


@pytest.fixture
def completed_run(tmp_path):
    root=tmp_path/'run';root.mkdir()
    y=np.tile(np.arange(5),2);m=classification_metrics(y,np.eye(5)[y])
    m.update(complete_partition=True,loss=.1)
    parts={};summary={};metrics={}
    for part,g in [('train','train_group'),('validation','validation_group')]:
        parts[part]=[g];summary[part]={'class_counts':dict(zip('NSVFQ',[2]*5))}
        metrics[part]={**m,'per_group':{g:dict(m)}}
    plan={'summary':summary,'split':{'groups':parts}}
    row={'epoch':1,'train':metrics['train'],'validation':metrics['validation']}
    (root/'epoch_001_metrics.json').write_text(json.dumps(row))
    cp=root/'epoch_001.pt';cp.write_bytes(b'checksum-only-fixture')
    cp.with_suffix('.sha256').write_text(sha256_file(cp))
    (root/'result.json').write_text(json.dumps({'completed_epochs':1,'best_epoch':1,'best_validation_macro_f1':1.}))
    (root/'training_plan.json').write_text(json.dumps(plan))
    return root


def test_validated_history_consistency(completed_run):
    result,plan,history,best=validated_history(completed_run)
    assert result['best_epoch']==best['epoch']==1 and len(history)==1
    assert best['validation']['macro_f1']==1


@pytest.mark.parametrize('change',['support','macro','group','coverage','checksum','best_epoch'])
def test_history_tampering_rejected(completed_run,change):
    path=completed_run/'epoch_001_metrics.json';row=json.loads(path.read_text())
    if change=='support':row['validation']['beat_count']=9
    elif change=='macro':row['validation']['macro_f1']=.5
    elif change=='group':row['validation']['per_group']={'test_group':next(iter(row['validation']['per_group'].values()))}
    elif change=='coverage':row['validation']['complete_partition']=False
    elif change=='checksum':(completed_run/'epoch_001.pt').write_bytes(b'corrupt')
    elif change=='best_epoch':
        p=completed_run/'result.json';d=json.loads(p.read_text());d['best_epoch']=2;p.write_text(json.dumps(d))
    path.write_text(json.dumps(row))
    with pytest.raises((ValueError,AssertionError)):validated_history(completed_run)
