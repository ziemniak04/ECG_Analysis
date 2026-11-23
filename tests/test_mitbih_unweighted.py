"""Controlled loss-only experiment and validation audit mechanics, synthetic data."""
from dataclasses import replace,asdict
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from test_mitbih_setup import fixture_export
from src.mitbih_unweighted import controlled_config,loss_summary,assert_validation_batch,evaluate
from src.mitbih_training import MITBIHTrainingConfig
from src.mitbih_torch import MITBIHBeatDataset
from src.models.mitbih_cnn import MITBIHBeatCNN
from src.prepare_dataset import sha256_file


def test_exact_loss_only_configuration():
    base=MITBIHTrainingConfig();candidate=replace(base,loss_strategy='unweighted')
    controlled_config(asdict(base),candidate)
    assert asdict(base)['loss_strategy']=='sqrt_balanced'


@pytest.mark.parametrize('kwargs',[dict(seed=43),dict(batch_size=64),dict(learning_rate=.002),dict(loss_strategy='balanced')])
def test_extra_experimental_changes_rejected(kwargs):
    base=MITBIHTrainingConfig();candidate=replace(base,loss_strategy='unweighted')
    with pytest.raises(ValueError,match='Only loss_strategy'):controlled_config(asdict(base),replace(candidate,**kwargs))


def test_exact_weighted_loss_matches_pytorch_and_unweighted():
    logits=torch.tensor([[2.,1,0,-1,-2],[0,1,2,3,4],[1,1,1,1,1],[5.,0,0,0,0]])
    labels=torch.tensor([0,4,3,1]);w=torch.tensor([.2,1.,.7,2.3,.8])
    ce=torch.nn.functional.cross_entropy(logits,labels,reduction='none').numpy()
    for weights in (None,w.numpy()):
        actual=loss_summary(ce,labels.numpy(),weights)
        ref=torch.nn.CrossEntropyLoss(weight=None if weights is None else w)(logits,labels)
        assert actual['loss']==pytest.approx(float(ref),rel=1e-6)
    # Summing batch numerators/denominators matches the whole-population mean.
    a=loss_summary(ce[:3],labels[:3].numpy(),w.numpy());b=loss_summary(ce[3:],labels[3:].numpy(),w.numpy())
    total=loss_summary(ce,labels.numpy(),w.numpy())
    assert (a['numerator']+b['numerator'])/(a['denominator']+b['denominator'])==pytest.approx(total['loss'])
    with pytest.raises(ValueError):loss_summary([float('nan')],np.array([0]),None)
    with pytest.raises(ValueError):loss_summary([1.],np.array([5]),None)


def test_no_test_dataset_or_nonvalidation_batch():
    with pytest.raises(PermissionError):evaluate('never_read',SimpleNamespace(partition='test'),{},None)
    split={'groups':{'validation':['val']},'summary':{'validation':{'record_ids':['104']}}}
    with pytest.raises(PermissionError):assert_validation_batch({'patient_group':['test'],'record_id':['104']},split)
    with pytest.raises(PermissionError):assert_validation_batch({'patient_group':['val'],'record_id':['test_record']},split)


def test_frozen_full_validation_metadata_and_parameters_preserved(fixture_export,tmp_path):
    torch.set_num_threads(2)
    index,split=fixture_export;model=MITBIHBeatCNN();cp=tmp_path/'best.pt'
    torch.save(dict(schema='mitbih-best-model-v1',class_names=list('NSVFQ'),model_state_dict=model.state_dict()),cp)
    digest=sha256_file(cp);cp.with_suffix('.sha256').write_text(digest)
    dataset=MITBIHBeatDataset(index,split,partition='validation',max_cached_shards=2)
    metrics,groups,metadata,y,p,ce=evaluate(cp,dataset,split,np.array([.2,1,.7,2,.8]))
    assert metrics['parameters_unchanged'] and sha256_file(cp)==digest
    assert y.shape==(len(dataset),) and p.shape==(len(dataset),5) and len(metadata)==len(dataset)
    assert len({r['source_index'] for r in metadata})==len(dataset)
    assert set(groups)==set(split['groups']['validation'])
    assert set(index.read_records)==set(split['summary']['validation']['record_ids'])
    assert metrics['unweighted_validation_ce']['loss']==pytest.approx(float(ce.mean()))
    assert all(r['original_symbol'] in ('N','A','V','F','/') for r in metadata)
    assert metrics['classes_with_nonzero_recall']==sum(r['recall']>0 for r in metrics['per_class'])


def test_frozen_training_loss_audit_never_updates_or_reads_test(fixture_export,tmp_path):
    from examples.compare_mitbih_unweighted import frozen_train_losses
    index,split=fixture_export;model=MITBIHBeatCNN();path=tmp_path/'frozen.pt'
    torch.save({'model_state_dict':model.state_dict()},path)
    ds=MITBIHBeatDataset(index,split,partition='train',max_cached_shards=4)
    result=frozen_train_losses(path,ds,split,np.ones(5))
    assert result['beat_count']==len(ds) and result['parameters_unchanged']
    assert result['unweighted_ce']['loss']==result['sqrt_weighted_ce']['loss']
    assert set(index.read_records)<=set(split['summary']['train']['record_ids'])
    ds.partition='validation'
    with pytest.raises(PermissionError):frozen_train_losses('never_opened',ds,split,np.ones(5))
