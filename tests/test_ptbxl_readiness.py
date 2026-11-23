"""Checkpoint-A guards/projections and small controlled CPU probes; no full training."""
from dataclasses import replace
from pathlib import Path
import json
import math
import time
import numpy as np
import pytest
import torch
from test_ptbxl_training import fixture_index
from src.ptbxl_protocol import training_class_counts,training_class_weights,file_sha256
from src.ptbxl_training import build_plan,TrainingConfig,train_model
from src.ptbxl_readiness import (full_configuration,validate_full_configuration,TimedBatches,
    summarize_timings,project_epoch,recommend_case,benchmark_case,storage_projection,run_readiness)


def test_full_configuration_uses_entire_validation_and_explicit_approval(tmp_path):
    config=full_configuration()
    validate_full_configuration(config)
    assert config.epochs==15 and config.patience==4 and config.seed==42
    assert config.batch_size==32 and config.learning_rate==.001
    assert config.class_weighted_loss and config.device=='cpu'
    assert config.train_per_class is config.validation_per_class is None
    assert config.max_train_batches is config.max_validation_batches is None
    with pytest.raises(PermissionError):train_model(config,tmp_path/'forbidden')
    assert not (tmp_path/'forbidden').exists()


@pytest.mark.parametrize('kwargs',[{'validation_per_class':16},{'max_validation_batches':4},
                                   {'train_per_class':32},{'max_train_batches':8},{'device':'cuda'}])
def test_readiness_rejects_truncated_or_wrong_device_full_configuration(kwargs):
    with pytest.raises(ValueError):validate_full_configuration(replace(full_configuration(),**kwargs))


def test_full_plan_on_controlled_index_preserves_all_fold9_and_training_weights(fixture_index):
    plan=build_plan(fixture_index,full_configuration(batch_size=4))
    assert plan['partitions']['train']['records']==8 and plan['partitions']['validation']['records']==4
    assert plan['partitions']['validation']['global_indices']==list(fixture_index.partitions.validation)
    assert plan['partitions']['validation']['folds']==[9]
    assert plan['class_weights']==[1.,1.,1.,1.]
    with pytest.raises(ValueError):training_class_weights(fixture_index,fixture_index.partitions.test)


def test_timed_loader_profiles_exactly_requested_batches():
    source=iter([{'y':torch.arange(4)} for _ in range(3)])
    measured=TimedBatches(source,2)
    assert len(list(measured))==2 and len(measured.timings)==2
    assert next(source)['y'].shape==(4,)
    summary=summarize_timings(measured.timings)
    assert summary['records']==8 and summary['records_per_second']>0
    with pytest.raises(ValueError):summarize_timings([])


def test_epoch_projection_is_explicit_measured_extrapolation():
    train={'mean_batch_seconds':2.,'min_batch_seconds':1.,'max_batch_seconds':3.,
           'records':192,'metric_tail_seconds':0.,'loader_setup_seconds':0.}
    val={**train,'mean_batch_seconds':1.,'min_batch_seconds':.5,'max_batch_seconds':1.5}
    result=project_epoch(15689,1958,32,train,val)
    assert result['training_batches']==491 and result['validation_batches']==62
    assert result['estimated_epoch_seconds']==491*2+62
    assert result['planning_range_seconds']==[491+31,(491*3+62*1.5)*1.25]
    assert result['checkpoint_and_startup_time_included'] is False


def test_case_recommendation_uses_only_speed_and_prefers32_within10percent():
    cases=[{'batch_size':16,'cpu_threads':1,'projection':{'estimated_epoch_seconds':100}},
           {'batch_size':32,'cpu_threads':1,'projection':{'estimated_epoch_seconds':108}},
           {'batch_size':32,'cpu_threads':2,'projection':{'estimated_epoch_seconds':105}}]
    assert recommend_case(cases)==cases[1]
    with pytest.raises(ValueError):recommend_case([])


def test_limited_probe_never_reads_test_or_saves_model(fixture_index,monkeypatch,tmp_path):
    original=fixture_index.load_numpy;reads=[]
    def guarded(i,**kwargs):
        assert fixture_index.entries[i].strat_fold!=10
        reads.append(i);return original(i,**kwargs)
    monkeypatch.setattr(fixture_index,'load_numpy',guarded)
    result=benchmark_case(fixture_index,4,1,timed_batches=1,warmup_batches=0)
    assert len(reads)==8
    assert result['training_optimizer_updates']==1 and result['training_completed_epochs']==0
    assert result['validation']['folds']==[9]
    assert result['train']['records']==result['validation']['records']==4
    assert result['parameter_count']==74116 and result['finite_parameters']
    assert result['sampled_peak_rss_bytes']>0
    assert result['train']['quality_metrics_reported'] is False
    for args in ({'timed_batches':9},{'warmup_batches':3}):
        with pytest.raises(ValueError):benchmark_case(fixture_index,4,1,**args)
    with pytest.raises(ValueError):benchmark_case(fixture_index,32,1)


def test_non_overwriting_readiness_before_data_access(tmp_path):
    out=tmp_path/'existing';out.mkdir();(out/'benchmark.json').write_text('keep')
    with pytest.raises(FileExistsError):run_readiness(out,tmp_path/'config.json')
    assert (out/'benchmark.json').read_text()=='keep'
    config=tmp_path/'config.json';config.write_text('keep')
    with pytest.raises(FileExistsError):run_readiness(tmp_path/'new',config)


def test_actual_training_counts_and_weights_remain_train_only():
    from src.ptbxl_protocol import PTBXLExportIndex
    index=PTBXLExportIndex()
    config=TrainingConfig(**json.loads(Path('configs/ptbxl_phase18_full_v1.json').read_text()))
    validate_full_configuration(config);plan=build_plan(index,config)
    np.testing.assert_array_equal(training_class_counts(index,index.partitions.train),[13354,1189,643,503])
    np.testing.assert_allclose(plan['class_weights'],[.29371348,3.29878044,6.09992218,7.79771376],rtol=1e-6)
    assert plan['partitions']['train']['records']==15689
    assert plan['partitions']['validation']['records']==1958
    assert plan['partitions']['validation']['batches_per_epoch']==math.ceil(1958/config.batch_size)
    assert plan['partitions']['validation']['maximum_records_per_epoch']==1958
    assert plan['partitions']['validation']['global_indices']==list(index.partitions.validation)
    assert plan['test_fold_used'] is False and plan['patient_overlap'] is False


def test_checkpoint_io_probe_and_storage_projection_preserve_existing_checkpoint(fixture_index,tmp_path):
    from pathlib import Path
    baseline=Path('results/model_training/phase17_baseline')
    original=file_sha256(baseline/'checkpoints/epoch_002.pt')
    plan=build_plan(fixture_index,full_configuration(batch_size=4))
    sizing=storage_projection(tmp_path,plan)
    assert sizing['probe_reload_identical'] is True
    assert sizing['measured_checkpoint_write_seconds']>0 and sizing['measured_checkpoint_load_seconds']>0
    assert sizing['recommended_disk_reserve_bytes']>=2**30
    assert sizing['estimated_full_python_metric_history_bytes']>0
    assert file_sha256(baseline/'checkpoints/epoch_002.pt')==original
    with pytest.raises(FileExistsError):storage_projection(tmp_path,plan)


def test_resume_with_planned_weighting_and_two_cpu_threads(fixture_index,tmp_path):
    """Small synthetic fixture verifies the planned execution settings, not ECG quality."""
    from src.ptbxl_training import load_checkpoint
    config=TrainingConfig(batch_size=4,cpu_threads=2,class_weighted_loss=True,
        train_per_class=2,validation_per_class=1,max_train_batches=2,max_validation_batches=1)
    whole=tmp_path/'whole';train_model(config,whole,index=fixture_index)
    first=tmp_path/'first';train_model(config,first,index=fixture_index,epoch_budget=1)
    resumed=tmp_path/'resumed'
    result=train_model(config,resumed,index=fixture_index,resume_from=first/'checkpoints/epoch_001.pt')
    a=load_checkpoint(whole/'checkpoints/epoch_002.pt');b=load_checkpoint(resumed/'checkpoints/epoch_002.pt')
    assert result['completed_epochs']==2 and result['checkpoint_reload_identical_logits']
    assert a['plan']['configuration']['cpu_threads']==2
    assert a['plan']['class_weights'] is not None
    for key in a['model_state_dict']:
        torch.testing.assert_close(a['model_state_dict'][key],b['model_state_dict'][key],rtol=0,atol=0)
    assert a['history'][1]['validation']['macro_f1']==b['history'][1]['validation']['macro_f1']
