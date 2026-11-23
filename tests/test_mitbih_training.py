"""Training/readiness tests use synthetic shards; real benchmark is separate."""
from dataclasses import replace
import json
import numpy as np
import pytest
import torch
from test_mitbih_setup import fixture_export
from src.mitbih_training import (MITBIHTrainingConfig, loss_weights, build_plan, make_loader,
    epoch_pass, train_model, load_checkpoint)
from src.mitbih_torch import MITBIHBeatDataset
from src.mitbih_metrics import classification_metrics
from src.models.mitbih_cnn import MITBIHBeatCNN
from src.prepare_dataset import sha256_file
from src.mitbih_readiness import benchmark, preload, verify_frozen_protocol


@pytest.mark.parametrize('kwargs', [dict(batch_size=0),dict(batch_size=True),dict(learning_rate=0),
    dict(learning_rate=float('nan')),dict(seed=-1),dict(device='cuda'),dict(num_workers=2),
    dict(loss_strategy='other'),dict(patience=0),dict(validation_cache_shards=0),dict(min_delta=-1)])
def test_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError): MITBIHTrainingConfig(**kwargs)


def test_metrics_known_confusion_matrix_and_rates():
    true = np.array([0,0,1,1,2,2,3,3,4,4]); pred = np.array([0,1,1,1,2,3,3,3,4,0])
    metrics = classification_metrics(true,np.eye(5)[pred])
    assert metrics['accuracy'] == .7
    np.testing.assert_array_equal(metrics['confusion_matrix'], [[1,1,0,0,0],[0,2,0,0,0],[0,0,1,1,0],[0,0,0,2,0],[1,0,0,0,1]])
    n = metrics['per_class'][0]
    assert (n['tp'],n['fp'],n['fn'],n['tn']) == (1,1,1,7)
    assert n['precision']==n['recall']==n['f1']==.5 and n['specificity']==7/8
    assert metrics['macro_f1']==pytest.approx(np.mean([.5,.8,2/3,.8,2/3]))
    # Two probability thresholds (1 and 0): tie handling includes all remaining scores.
    assert n['auprc']==pytest.approx(.5*.5+.5*.2)


def test_perfect_and_absent_class_metrics_are_honest():
    y = np.tile(np.arange(5),2)
    m = classification_metrics(y,np.eye(5)[y]); assert m['macro_auprc']==m['macro_f1']==1
    m = classification_metrics(np.array([0,0]),np.eye(5)[[0,0]])
    assert m['macro_auprc'] is None and m['macro_f1'] is None
    assert m['per_class'][1]['recall'] is None and m['per_class'][1]['precision'] is None
    assert m['per_class'][0]['specificity'] is None


@pytest.mark.parametrize('y,p', [(np.array([5]),np.ones((1,5))/5), (np.array([0.]),np.ones((1,5))/5),
    (np.array([0]),np.ones((1,4))/4),(np.array([0]),np.zeros((1,5))),
    (np.array([0]),np.full((1,5),np.nan))])
def test_metrics_reject_invalid_inputs(y,p):
    with pytest.raises(ValueError): classification_metrics(y,p)


def test_weights_exact_training_only_and_square_root(fixture_export):
    index, split = fixture_export
    counts=np.array([split['summary']['train']['class_counts'][c] for c in 'NSVFQ'])
    expected=counts.sum()/(5*counts)
    np.testing.assert_allclose(loss_weights(index,split,'balanced'),expected)
    np.testing.assert_allclose(loss_weights(index,split,'sqrt_balanced'),np.sqrt(expected)/np.sqrt(expected).mean())
    assert loss_weights(index,split,'unweighted') is None and not index.read_records
    altered=json.loads(json.dumps(split));altered['summary']['validation']['class_counts']['F']+=1
    with pytest.raises(ValueError):loss_weights(index,altered,'balanced')


def test_loader_determinism_and_complete_coverage(fixture_export):
    index,split=fixture_export;dataset=MITBIHBeatDataset(index,split,partition='train')
    config=MITBIHTrainingConfig(batch_size=8)
    def order(epoch):return [int(i) for b in make_loader(dataset,config,epoch) for i in b['global_source_index']]
    a=order(1);assert a==order(1) and a!=order(2) and len(a)==len(set(a))==len(dataset)
    dataset.partition='test'
    with pytest.raises(PermissionError):make_loader(dataset,config,1)


def test_epoch_loss_exact_and_no_test_or_leaked_groups(fixture_export):
    index,split=fixture_export;dataset=MITBIHBeatDataset(index,split,partition='validation')
    config=MITBIHTrainingConfig(batch_size=3);model=MITBIHBeatCNN()
    weights=torch.tensor([.2,.5,1.,2.,3.])
    result=epoch_pass(model,make_loader(dataset,config,1),split,partition='validation',weights=weights)
    assert result['complete_partition'] and result['beat_count']==len(dataset)
    assert set(result['per_group'])==set(split['groups']['validation'])
    assert result['loss']==result['loss_numerator']/result['loss_denominator']
    expected=sum(split['summary']['validation']['class_counts'][c]*float(weights[i]) for i,c in enumerate('NSVFQ'))
    assert result['loss_denominator']==pytest.approx(expected)
    assert result['inference_ms_per_beat']>=0
    with pytest.raises(PermissionError):epoch_pass(model,[],split,partition='test')
    batch=next(iter(make_loader(dataset,config,1)));batch['patient_group'][0]=split['groups']['test'][0]
    with pytest.raises(RuntimeError,match='leakage'):epoch_pass(model,[batch],split,partition='validation',batch_cap=1)


def test_full_training_gate_before_any_io(tmp_path):
    with pytest.raises(PermissionError):train_model(MITBIHTrainingConfig(),tmp_path/'no')
    assert not (tmp_path/'no').exists()


def test_resume_exact_rng_optimizer_best_state_and_safe_outputs(fixture_export,tmp_path):
    index,split=fixture_export;config=MITBIHTrainingConfig(epochs=3,batch_size=16)
    full=tmp_path/'full';part=tmp_path/'part';resumed=tmp_path/'resumed'
    a=train_model(config,full,index=index,split=split,allow_full_training=True)
    b=train_model(config,part,index=index,split=split,allow_full_training=True,epoch_budget=1)
    c=train_model(config,resumed,index=index,split=split,allow_full_training=True,resume_from=b['last_checkpoint'])
    x=torch.load(a['last_checkpoint'],weights_only=True);z=torch.load(c['last_checkpoint'],weights_only=True)
    for k,v in x['model_state_dict'].items():torch.testing.assert_close(v,z['model_state_dict'][k],rtol=0,atol=0)
    assert a['best_validation_macro_f1']==c['best_validation_macro_f1'] and not c['final_test_used']
    for result in (a,c):
        cp=torch.load(result['last_checkpoint'],weights_only=True)
        assert all(r['validation']['beat_count']==split['summary']['validation']['beats'] for r in cp['history'])
        assert all(r['validation']['complete_partition'] for r in cp['history'])
    assert set(index.read_records)<=set(split['summary']['train']['record_ids']+split['summary']['validation']['record_ids'])
    with pytest.raises(FileExistsError):train_model(config,full,index=index,split=split,allow_full_training=True)
    plan=build_plan(index,split,config)
    with pytest.raises(ValueError,match='differs'):load_checkpoint(b['last_checkpoint'],build_plan(index,split,replace(config,loss_strategy='unweighted')))
    path=part/'epoch_001.pt'
    malformed=torch.load(path,weights_only=True)
    next(iter(malformed['optimizer_state_dict']['state'].values()))['exp_avg'].fill_(float('nan'))
    torch.save(malformed,path);path.with_suffix('.sha256').write_text(sha256_file(path))
    with pytest.raises(ValueError,match='optimizer state'):load_checkpoint(path,plan)
    path.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='SHA256'):load_checkpoint(path,plan)


def test_early_stop_and_no_resume_after_stopping(fixture_export,tmp_path,monkeypatch):
    import src.mitbih_training as module
    index,split=fixture_export;original=module.epoch_pass
    def constant(*args,**kwargs):
        result=original(*args,**kwargs)
        if kwargs['partition']=='validation': result['macro_f1']=.1
        return result
    monkeypatch.setattr(module,'epoch_pass',constant)
    config=MITBIHTrainingConfig(epochs=5,patience=1,batch_size=16)
    r=train_model(config,tmp_path/'stop',index=index,split=split,allow_full_training=True)
    assert r['status']=='early_stopped' and r['completed_epochs']==2 and r['best_epoch']==1
    with pytest.raises(ValueError,match='complete or early-stopped'):
        load_checkpoint(r['last_checkpoint'],build_plan(index,split,config))


def test_interrupted_partial_epoch_keeps_last_completed_checkpoint(fixture_export,tmp_path,monkeypatch):
    import src.mitbih_training as module
    index,split=fixture_export;original=module.epoch_pass;calls=[]
    def interrupt(*args,**kwargs):
        calls.append(kwargs['partition'])
        if len(calls)==3: raise KeyboardInterrupt()
        return original(*args,**kwargs)
    monkeypatch.setattr(module,'epoch_pass',interrupt)
    r=train_model(MITBIHTrainingConfig(epochs=3,batch_size=16),tmp_path/'interrupt',index=index,split=split,allow_full_training=True)
    assert r['status']=='interrupted_partial_epoch_discarded' and r['completed_epochs']==1
    assert r['last_checkpoint'].endswith('epoch_001.pt')


def test_preload_bounded_partition_and_no_final_test(fixture_export):
    index,split=fixture_export;dataset=MITBIHBeatDataset(index,split,max_cached_shards=1)
    r=preload(dataset);assert r['cache_entries']==1 and r['shards']==4 and r['cache_bytes']>0
    dataset.partition='test'
    with pytest.raises(PermissionError):preload(dataset)


def test_benchmark_caps_rejected_before_any_dataset_io(tmp_path):
    with pytest.raises(ValueError):benchmark(tmp_path/'bad',timed_batches=7)
    with pytest.raises(ValueError):benchmark(tmp_path/'bad',batch_sizes=(128,))


def test_real_frozen_split_metadata_only():
    from src.mitbih_protocol import MITBIHExportIndex,load_split
    index=MITBIHExportIndex();split=load_split(index)
    summary=verify_frozen_protocol(index,split)
    assert [summary[p]['beats'] for p in ('train','validation','test')]==[62926,23134,23392]
    assert index.read_records==[]
