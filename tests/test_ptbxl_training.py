"""Small controlled test fixtures validate mechanics, never clinical performance."""
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from src.ptbxl_protocol import PTBXLExportIndex,file_sha256
from src.ptbxl_training import (TrainingConfig,build_plan,select_indices,verify_selection,
    train_model,load_checkpoint,epoch_pass,prepare_device)
from src.models.ptbxl_cnn import PTBXLRhythmCNN


@pytest.fixture
def fixture_index(tmp_path):
    root=tmp_path/'fixture';(root/'recordings').mkdir(parents=True)
    rows=[];classes=['SR','AFIB','STACH','SBRAD'];rng=np.random.default_rng(7)
    for i in range(16):
        label=i%4;code=classes[label];eid=90001+i;pid=5001+i
        fold=1 if i<8 else (9 if i<12 else 10)
        identity={'ecg_id':eid,'patient_id':pid,'strat_fold':fold,'label':code,'original_scp_statements':{code:0.}}
        path=root/f'recordings/{eid:05d}_rhythm.npz'
        np.savez_compressed(path,X=rng.normal(size=(12,1000)).astype(np.float32),y=np.array(label,dtype=np.int64),
            ecg_id=np.array(eid,dtype=np.int64),patient_id=np.array(pid,dtype=np.int64),strat_fold=np.array(fold,dtype=np.int64),
            rhythm_code=np.array(code),original_scp_codes=np.array(json.dumps({code:0.})),original_metadata=np.array(json.dumps(identity)),
            channel_names=np.array(['I','II','III','AVR','AVL','AVF','V1','V2','V3','V4','V5','V6']),
            class_names=np.array(classes),sampling_frequency=np.array(100.),preprocessing_configuration=np.array('{}'),source_hashes=np.array('{}'))
        rows.append({'ecg_id':eid,'patient_id':pid,'strat_fold':fold,'label':code,'file':str(path.relative_to(root)),
                     'sha256':file_sha256(path),'status':'exported'})
    manifest={'status':'complete','records':rows,'selected_records':16,'successful_records':16,
              'successful_patients':16,'failed_records':0,'not_attempted_records':0}
    path=root/'manifest.json';path.write_text(json.dumps(manifest))
    return PTBXLExportIndex(root,path,require_full_cohort=False,verify_source_metadata=False)


def config(**kwargs):
    return TrainingConfig(epochs=2,batch_size=4,train_per_class=2,validation_per_class=1,
        max_train_batches=2,max_validation_batches=1,**kwargs)


def test_deterministic_class_aware_subset_and_no_test_access(fixture_index):
    a=select_indices(fixture_index,'train',1,42);b=select_indices(fixture_index,'train',1,42)
    assert a==b and sorted(fixture_index.entries[i].label for i in a)==[0,1,2,3]
    assert a==tuple(sorted(a))
    with pytest.raises(ValueError):select_indices(fixture_index,'test',1,42)
    with pytest.raises(ValueError):select_indices(fixture_index,'validation',2,42)
    with pytest.raises(ValueError):verify_selection(fixture_index,a,(12,))
    plan=build_plan(fixture_index,config())
    assert plan['partitions']['train']['class_counts']==[2,2,2,2]
    assert plan['partitions']['validation']['folds']==[9] and plan['test_fold_used'] is False


def test_configuration_short_limits_and_full_opt_in(tmp_path,monkeypatch):
    for kwargs in ({'epochs':3},{'train_per_class':None},{'max_train_batches':None},{'learning_rate':float('nan')},{'patience':0}):
        with pytest.raises(ValueError):TrainingConfig(**kwargs)
    with pytest.raises(PermissionError):train_model(TrainingConfig(mode='full'),tmp_path/'no')
    assert not (tmp_path/'no').exists()
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    with pytest.raises(RuntimeError,match='unavailable'):prepare_device(TrainingConfig(device='cuda'))


def test_optional_weights_come_from_actual_training_subset(fixture_index):
    plan=build_plan(fixture_index,config(class_weighted_loss=True))
    assert plan['class_weights']==[1.,1.,1.,1.]
    assert plan['class_weight_source']=='selected_training_indices_only'


def test_short_real_model_training_checkpoint_reload_and_no_test_reads(fixture_index,tmp_path,monkeypatch):
    reads=[];original=fixture_index.load_numpy
    def observed(i,**kwargs):
        assert fixture_index.entries[i].strat_fold!=10
        reads.append(i);return original(i,**kwargs)
    monkeypatch.setattr(fixture_index,'load_numpy',observed)
    out=tmp_path/'run';result=train_model(config(),out,index=fixture_index)
    assert result['status']=='completed' and result['optimizer_steps']==4
    assert result['parameters_changed_this_run'] and result['checkpoint_reload_identical_logits']
    assert result['test_fold_used'] is False and reads
    history=[json.loads((out/f'epoch_{e:03d}_metrics.json').read_text()) for e in (1,2)]
    assert all(np.isfinite(r['train']['loss']) and np.isfinite(r['validation']['loss']) for r in history)
    assert all(r['train']['finite_gradients'] for r in history)
    assert all(r['validation']['record_count']==4 for r in history)
    cp=load_checkpoint(out/'checkpoints/epoch_002.pt')
    assert cp['completed_epoch']==2 and cp['optimizer_steps']==4
    assert cp['best_score']==max(r['validation']['macro_f1'] for r in history)
    assert cp['best_epoch']==min(r['epoch'] for r in history if r['validation']['macro_f1']==cp['best_score'])
    model=PTBXLRhythmCNN();model.load_state_dict(cp['model_state_dict'])
    assert model.parameter_count()==74116
    for file in ('loss_curves.png','validation_macro_f1.png','validation_confusion_matrix.png','reproducibility.json'):
        assert (out/file).is_file()
    with pytest.raises(FileExistsError):train_model(config(),out,index=fixture_index)


def test_resume_identical_epoch_boundary_and_configuration_guard(fixture_index,tmp_path):
    cfg=config()
    uninterrupted=tmp_path/'whole';train_model(cfg,uninterrupted,index=fixture_index)
    first=tmp_path/'first';one=train_model(cfg,first,index=fixture_index,epoch_budget=1)
    assert one['status']=='invocation_epoch_budget_reached' and one['completed_epochs']==1
    second=tmp_path/'resumed';result=train_model(cfg,second,index=fixture_index,resume_from=first/'checkpoints/epoch_001.pt')
    assert result['completed_epochs']==2 and result['epochs_executed_this_run']==1
    a=load_checkpoint(uninterrupted/'checkpoints/epoch_002.pt');b=load_checkpoint(second/'checkpoints/epoch_002.pt')
    for key in a['model_state_dict']:
        torch.testing.assert_close(a['model_state_dict'][key],b['model_state_dict'][key],rtol=0,atol=0)
    assert a['history'][1]['train']['loss']==b['history'][1]['train']['loss']
    assert a['history'][1]['validation']['macro_f1']==b['history'][1]['validation']['macro_f1']
    with pytest.raises(ValueError,match='differs'):
        train_model(replace(cfg,seed=99),tmp_path/'bad',index=fixture_index,resume_from=first/'checkpoints/epoch_001.pt')
    assert not (tmp_path/'bad').exists()


def test_validation_uses_eval_and_preserves_batchnorm_buffers(fixture_index):
    from src.ptbxl_training import make_loader
    cfg=config();torch.set_num_threads(1);model=PTBXLRhythmCNN()
    before={k:v.clone() for k,v in model.state_dict().items()}
    loader=make_loader(fixture_index,fixture_index.partitions.validation,'validation',cfg,1)
    metrics=epoch_pass(model,loader,torch.device('cpu'),partition='validation')
    assert metrics['inference_ms_per_record']>=0 and metrics['strat_folds']==[9]*4
    assert model.training is False
    for key,value in model.state_dict().items():torch.testing.assert_close(before[key],value,rtol=0,atol=0)
    with pytest.raises(ValueError):epoch_pass(model,loader,torch.device('cpu'),partition='test')


def test_weighted_epoch_loss_matches_exact_weight_denominator(fixture_index):
    from src.ptbxl_training import make_loader
    cfg=config();model=PTBXLRhythmCNN();weights=torch.tensor([1.,2.,3.,4.])
    loader=make_loader(fixture_index,fixture_index.partitions.validation,'validation',cfg,1)
    batch=next(iter(loader));model.eval()
    with torch.no_grad():expected=torch.nn.CrossEntropyLoss(weight=weights)(model(batch['X']),batch['y']).item()
    metrics=epoch_pass(model,loader,torch.device('cpu'),partition='validation',weights=weights)
    assert metrics['loss']==pytest.approx(expected,rel=1e-6)
    assert metrics['loss_denominator']==10.


def test_early_stopping_on_validation_only(fixture_index,tmp_path,monkeypatch):
    import src.ptbxl_training as module
    original=module.epoch_pass
    def fixed_score(*args,**kwargs):
        result=original(*args,**kwargs)
        if kwargs['partition']=='validation':result['macro_f1']=.25
        return result
    monkeypatch.setattr(module,'epoch_pass',fixed_score)
    result=train_model(config(patience=1),tmp_path/'early',index=fixture_index)
    assert result['status']=='early_stopped' and result['best_epoch']==1
    assert load_checkpoint(tmp_path/'early/checkpoints/epoch_002.pt')['early_stopped']


def test_interruption_records_last_complete_epoch_for_resume(fixture_index,tmp_path,monkeypatch):
    import src.ptbxl_training as module
    original=module.epoch_pass;calls=[]
    def interrupted(*args,**kwargs):
        if kwargs['partition']=='train':
            calls.append(1)
            if len(calls)==2:raise KeyboardInterrupt()
        return original(*args,**kwargs)
    monkeypatch.setattr(module,'epoch_pass',interrupted)
    out=tmp_path/'interrupted';result=train_model(config(),out,index=fixture_index)
    assert result['status']=='interrupted_partial_epoch_discarded'
    assert result['completed_epochs']==1
    assert result['last_completed_checkpoint'].endswith('epoch_001.pt')
    assert (out/'checkpoints/epoch_001.pt').is_file()
    assert not (out/'checkpoints/epoch_002.pt').exists()


def test_batch_cap_does_not_fetch_extra_records(fixture_index,monkeypatch):
    from src.ptbxl_training import make_loader
    cfg=config();model=PTBXLRhythmCNN();torch.set_num_threads(1)
    reads=[];original=fixture_index.load_numpy
    def observed(i,**kwargs):reads.append(i);return original(i,**kwargs)
    monkeypatch.setattr(fixture_index,'load_numpy',observed)
    loader=make_loader(fixture_index,fixture_index.partitions.train,'train',cfg,1)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    metrics=epoch_pass(model,loader,torch.device('cpu'),partition='train',optimizer=optimizer,batch_cap=1)
    assert metrics['batches']==1 and metrics['record_count']==4 and len(reads)==4
