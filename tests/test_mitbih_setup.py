"""MIT-BIH setup tests: synthetic shards except explicit TRAINING integration."""
from pathlib import Path
import json
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader
from src.mitbih_protocol import (MITBIHExportIndex,validate_split,training_class_weights,
    load_split,DEFAULT_SPLIT,inventory,propose_split)
from src.mitbih_torch import MITBIHBeatDataset,collate_mitbih
from src.mitbih_model_config import BeatCNNConfig
from src.models.mitbih_cnn import MITBIHBeatCNN
from src.prepare_dataset import sha256_file,FORMAT_VERSION,MAPPING_VERSION


@pytest.fixture
def fixture_export(tmp_path):
    root=tmp_path/'export';(root/'recordings').mkdir(parents=True)
    ids=['100','101','102','200','201','202','203','204']
    groups={rid:('fixture_201_202' if rid in ('201','202') else 'fixture_'+rid) for rid in ids}
    patient={'recordings':{rid:{'patient_group':g} for rid,g in groups.items()}}
    (root/'patient_groups.json').write_text(json.dumps(patient))
    rows=[];totals=np.zeros(5,dtype=int);start=0;rng=np.random.default_rng(10)
    for rid in ids:
        y=np.array([0]*5+[1,2,3,4],dtype=np.int64) if rid=='100' else np.tile(np.arange(5,dtype=np.int64),2)
        n=len(y);symbols=np.asarray(['N','A','V','F','/'])[y]
        cfg={'mapping_version':MAPPING_VERSION,'normalization':{'scope':'recording','method':'zscore'},'sequence_length':252}
        path=root/f'recordings/{rid}.npz'
        np.savez_compressed(path,X=rng.normal(size=(n,2,252)).astype(np.float32),y=y,
            record_ids=np.full(n,rid),patient_groups=np.full(n,groups[rid]),
            annotation_positions=np.arange(n,dtype=np.int64)*360+100,
            annotation_indices=np.arange(n,dtype=np.int64),original_symbols=symbols,
            annotation_aux_notes=np.full(n,''),class_names=np.array(list('NSVFQ')),
            sampling_frequency=np.array(360.),channel_names=np.array(['MLII','V1']),
            preprocessing_config=np.array(json.dumps(cfg)))
        counts=np.bincount(y,minlength=5);totals+=counts
        rows.append({'record_id':rid,'patient_group':groups[rid],'file':f'recordings/{rid}.npz',
            'count':n,'class_counts':dict(zip('NSVFQ',map(int,counts))),'shape':[n,2,252],
            'sampling_frequency':360.,'channel_names':['MLII','V1'],'sha256':sha256_file(path)})
    m={'status':'complete','format_version':FORMAT_VERSION,'class_names':list('NSVFQ'),
       'sequence_length':252,'channels':2,'sampling_frequency':360.,'mapping_version':MAPPING_VERSION,
       'patient_mapping_file':'patient_groups.json','artifact_sha256':{'patient_groups.json':sha256_file(root/'patient_groups.json')},
       'shards':rows,'record_count':8,'patient_group_count':7,'segment_count':int(totals.sum()),
       'class_counts':dict(zip('NSVFQ',map(int,totals)))}
    (root/'dataset_metadata.json').write_text(json.dumps(m))
    index=MITBIHExportIndex(root,require_full_cohort=False)
    split={'manifest_sha256':index.manifest_sha256,'class_names':list('NSVFQ'),
           'groups':{'train':['fixture_100','fixture_101','fixture_201_202'],
                     'validation':['fixture_102','fixture_200'],'test':['fixture_203','fixture_204']}}
    split['summary']=validate_split(index,split)
    return index,split


def test_index_metadata_only_shared_group_and_counts(fixture_export):
    index,split=fixture_export
    assert index.read_records==[] and len(index.groups)==7
    assert index.by_record['201'].patient_group==index.by_record['202'].patient_group
    assert split['summary']['train']['record_count']==4
    assert {'201','202'}<=set(split['summary']['train']['record_ids'])
    assert sum(r['beats'] for r in split['summary'].values())==index.segment_count
    json.dumps(split);json.dumps(inventory(index))  # native JSON counts, not np.int64


@pytest.mark.parametrize('change',['overlap','missing','hash','counts'])
def test_invalid_split_rejected(fixture_export,change):
    index,original=fixture_export;split=json.loads(json.dumps(original))
    if change=='overlap':split['groups']['validation'].append(split['groups']['train'][0])
    if change=='missing':split['groups']['test'].pop()
    if change=='hash':split['manifest_sha256']='bad'
    if change=='counts':split['summary']['train']['beats']+=1
    with pytest.raises(ValueError):validate_split(index,split)


def test_test_partition_blocked_before_read(fixture_export,monkeypatch):
    index,split=fixture_export
    def forbidden(*args,**kwargs):raise AssertionError('No waveform read permitted')
    monkeypatch.setattr(index,'load_shard',forbidden)
    with pytest.raises(PermissionError,match='test is locked'):
        MITBIHBeatDataset(index,split,partition='test')
    for ambiguous in ('false',1,None):
        with pytest.raises(ValueError,match='explicit boolean'):
            MITBIHBeatDataset(index,split,partition='test',allow_final_test=ambiguous)


def test_source_cannot_be_loaded_through_wrong_partition(fixture_export):
    index,split=fixture_export
    with pytest.raises(PermissionError):index.load_shard('203',split,'train')
    with pytest.raises(PermissionError):index.load_shard('203',split,'test')
    assert not index.read_records


def test_lazy_record_loading_cache_bound_copy_and_metadata(fixture_export):
    index,split=fixture_export;dataset=MITBIHBeatDataset(index,split,max_cached_shards=1)
    assert index.read_records==[]
    first=dataset[0];assert index.read_records==['100']
    assert first['X'].shape==(2,252) and first['X'].dtype==torch.float32
    assert first['y'].dtype==torch.long and first['y'].ndim==0
    assert first['record_id']=='100' and first['patient_group']=='fixture_100'
    assert first['original_symbol']=='N' and first['annotation_position']==100
    assert first['channel_names']==('MLII','V1') and first['sampling_frequency']==360
    expected=first['X'].clone();first['X'].zero_()
    torch.testing.assert_close(dataset[0]['X'],expected)
    assert index.read_records==['100']
    dataset[index.by_record['100'].count]
    assert index.read_records==['100','101'] and len(dataset.cache)==1
    dataset[0];assert index.read_records==['100','101','100']


def test_split_is_copied_not_mutable_by_caller(fixture_export):
    index,split=fixture_export;dataset=MITBIHBeatDataset(index,split)
    split['groups']['train'].clear()
    assert dataset[0]['patient_group']=='fixture_100'


@pytest.mark.parametrize('position',[-1,True,1.5,100000])
def test_invalid_indices(fixture_export,position):
    index,split=fixture_export;dataset=MITBIHBeatDataset(index,split)
    with pytest.raises(IndexError):dataset[position]


def test_missing_and_corrupt_shards_raise(fixture_export):
    index,split=fixture_export;dataset=MITBIHBeatDataset(index,split)
    path=index.root/index.by_record['100'].file
    path.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='SHA256'):dataset[0]
    path.unlink()
    with pytest.raises(FileNotFoundError):dataset[0]


def test_mutation_of_cached_shard_is_detected(fixture_export):
    index,split=fixture_export;dataset=MITBIHBeatDataset(index,split)
    dataset[0]
    path=index.root/index.by_record['100'].file
    path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='Cached shard changed'):dataset[0]


def test_original_symbol_must_match_integer_label(fixture_export):
    index,split=fixture_export;path=index.root/index.by_record['100'].file
    with np.load(path,allow_pickle=False) as f:arrays={key:f[key] for key in f.files}
    arrays['original_symbols'][0]='F'  # y=0 remains N; do not silently remap.
    np.savez_compressed(path,**arrays)
    manifest=json.loads(index.manifest_path.read_text());manifest['shards'][0]['sha256']=sha256_file(path)
    index.manifest_path.write_text(json.dumps(manifest))
    changed=MITBIHExportIndex(index.root,require_full_cohort=False)
    split['manifest_sha256']=changed.manifest_sha256
    with pytest.raises(ValueError,match='symbols and class labels'):
        MITBIHBeatDataset(changed,split)[0]


def test_collation_preserves_source_alignment(fixture_export):
    index,split=fixture_export;dataset=MITBIHBeatDataset(index,split)
    batch=next(iter(DataLoader(dataset,batch_size=6,collate_fn=collate_mitbih)))
    assert batch['X'].shape==(6,2,252) and batch['y'].shape==(6,)
    assert batch['original_symbol']==['N']*5+['A']
    assert batch['annotation_position'].tolist()==[100,460,820,1180,1540,1900]
    assert batch['channel_names']==[('MLII','V1')]*6


def test_training_weights_exact_and_validation_test_rejected(fixture_export):
    index,split=fixture_export;groups=split['groups']['train']
    counts=np.array(list(split['summary']['train']['class_counts'].values()))
    # dictionary order is N,S,V,F,Q in fixture and production contract
    expected=counts.sum()/(5*counts)
    np.testing.assert_allclose(training_class_weights(index,split,groups),expected,rtol=1e-7)
    for bad in (None,[],groups+[groups[0]],split['groups']['validation'],list(index.groups),groups[:-1]):
        with pytest.raises(ValueError):training_class_weights(index,split,bad)


def test_five_logits_finite_backward_update_and_parameter_count():
    torch.set_num_threads(1);torch.manual_seed(42);model=MITBIHBeatCNN()
    assert model.parameter_count()==17797==BeatCNNConfig().expected_parameter_count()
    optimizer=torch.optim.Adam(model.parameters(),lr=.001);x=torch.randn(5,2,252);y=torch.arange(5)
    before=[p.detach().clone() for p in model.parameters()]
    logits=model(x);assert logits.shape==(5,5) and torch.isfinite(logits).all()
    loss=torch.nn.CrossEntropyLoss()(logits,y);assert torch.isfinite(loss);loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert sum(float(p.grad.abs().sum()) for p in model.parameters())>0
    optimizer.step();assert any(not torch.equal(a,b) for a,b in zip(before,model.parameters()))


def test_model_input_contract_and_separate_ptb_architecture():
    model=MITBIHBeatCNN().eval()
    for x in (torch.zeros(1,12,1000),torch.zeros(1,2,251),torch.zeros(0,2,252)):
        with pytest.raises(ValueError):model(x)
    with pytest.raises(TypeError):model(torch.zeros(1,2,252,dtype=torch.float64))
    x=torch.zeros(1,2,252);x[0,0,0]=float('nan')
    with pytest.raises(ValueError):model(x)
    from src.models.ptbxl_cnn import PTBXLRhythmCNN
    assert PTBXLRhythmCNN().parameter_count()==74116


def test_real_frozen_protocol_and_training_interface_only():
    if not DEFAULT_SPLIT.exists():pytest.skip('Freeze Phase-20 split before actual-cohort integration')
    index=MITBIHExportIndex();split=load_split(index)
    assert [split['summary'][p]['group_count'] for p in ('train','validation','test')]==[29,9,9]
    validate_split(index,split)
    pair=index.by_record['201'].patient_group
    assert pair==index.by_record['202'].patient_group
    assert sum(pair in split['groups'][p] for p in ('train','validation','test'))==1
    assert sum(r['beats'] for r in split['summary'].values())==109452
    assert all(min(r['class_counts'].values())>0 for r in split['summary'].values())
    sample=MITBIHBeatDataset(index,split,partition='train')[0]
    assert sample['patient_group'] in split['groups']['train']
    assert sample['X'].shape==(2,252) and torch.isfinite(sample['X']).all()
    assert not set(index.read_records)&set(split['summary']['test']['record_ids'])


def test_group_proposal_is_deterministic_and_metadata_only(monkeypatch):
    index=MITBIHExportIndex()
    def forbidden(*args,**kwargs):raise AssertionError('Split design must never read ECG arrays')
    monkeypatch.setattr(index,'load_shard',forbidden)
    a=propose_split(index,candidates=256);b=propose_split(index,candidates=256)
    assert a==b and index.read_records==[]
    assert a['rare_class_anchors']['train']==b['rare_class_anchors']['train']
    assert all(min(r['supporting_groups'].values())>=2 for r in a['summary'].values())
    json.dumps(a)
