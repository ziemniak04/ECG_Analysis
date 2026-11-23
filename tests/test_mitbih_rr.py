"""RR-context engineering and CNN fusion tests; no held-out data access."""
from dataclasses import replace
import json
import numpy as np
import pytest
import torch
from src.mitbih_rr import rr_features,beat_ids,fit_training_transform,transform_rr
from src.mitbih_rr_config import RRFeatureConfig,RR_FEATURE_NAMES
from src.models.mitbih_cnn_rr import MITBIHBeatCNNRR
from src.models.mitbih_cnn import MITBIHBeatCNN
from src.mitbih_rr_dataset import MITBIHRRDataset


def test_manual_causal_features_and_first_last_beat():
    p=np.array([0,360,1080,1440],dtype=np.int64)
    raw,valid=rr_features(p,p,360)
    np.testing.assert_allclose(raw,[[0,0,0],[1,0,0],[2,1,2],[1,1.5,2/3]])
    np.testing.assert_array_equal(valid,[[0,0,0],[1,0,0],[1,1,1],[1,1,1]])
    assert valid[-1].all()  # No future/next beat is required.


def test_history_is_exactly_eight_intervals_and_excludes_current():
    p=np.r_[0,np.cumsum(np.arange(1,11)*360)].astype(np.int64)
    raw,valid=rr_features(p,p[-1:],360)
    np.testing.assert_allclose(raw,[[10,5.5,10/5.5]])


def test_future_invariance_and_record_reset():
    p=np.array([10,370,1090],dtype=np.int64)
    a,ma=rr_features(p,p,360);b,mb=rr_features(np.r_[p,2000,4000],p,360)
    np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(ma,mb)
    q=np.array([20,380,740],dtype=np.int64)
    c,mc=rr_features(q,q,360)
    assert not mc[0].any() and c[0].sum()==0
    with pytest.raises(ValueError):rr_features(np.r_[p,q],p,360)


def test_sampling_frequency_and_boundary_skipped_context():
    p=np.array([50,150,250],dtype=np.int64)
    raw,mask=rr_features(p,p[1:],100)
    assert raw[0,0]==1 and mask[0,0] and not mask[0,1]
    raw2,_=rr_features(p,p[1:],200)
    assert raw2[0,0]==.5


@pytest.mark.parametrize('p,t,fs',[
    ([0,0,360],[360],360),([360,0],[0],360),([-1,360],[360],360),
    ([0,360],[1],360),([0,360],[360],0),([0,360],[360],float('nan'))])
def test_invalid_timestamps_and_sampling_rejected(p,t,fs):
    with pytest.raises(ValueError):rr_features(np.array(p,dtype=np.int64),np.array(t,dtype=np.int64),fs)


def transform_fixture():
    p=np.array([0,360,1080,1440,1800],dtype=np.int64)
    raw,valid=rr_features(p,p,360)
    state=fit_training_transform(raw,valid,partition='train',patient_groups=['a','a','b','b','b'],
        expected_training_groups=['a','b'],manifest_sha256='a'*64)
    return raw,valid,state


def test_training_only_fit_valid_values_and_explicit_masks():
    raw,valid,state=transform_fixture()
    assert state['imputation_median'][0]==1 and state['valid_training_counts']==[4,3,3]
    out=transform_rr(raw,valid,state)
    assert out.dtype==np.float32 and out.shape==(5,6) and np.isfinite(out).all()
    np.testing.assert_array_equal(out[:,3:],valid)
    assert out[0,0]==pytest.approx((state['imputation_median'][0]-state['mean'][0])/state['scale'][0])
    old=json.dumps(state,sort_keys=True)
    other=raw.copy();other[valid]*=2
    transform_rr(other,valid,state)
    assert json.dumps(state,sort_keys=True)==old  # No refitting on validation/inference.
    for part in ('validation','test'):
        with pytest.raises(PermissionError):fit_training_transform(raw,valid,partition=part,patient_groups=['a']*5,expected_training_groups=['a'],manifest_sha256='a'*64)
    with pytest.raises(ValueError):fit_training_transform(raw,valid,partition='train',patient_groups=['val']*5,expected_training_groups=['a'],manifest_sha256='a'*64)


def test_constant_scale_finite_and_invalid_values_rejected():
    raw=np.ones((4,3));valid=np.ones((4,3),dtype=bool)
    s=fit_training_transform(raw,valid,partition='train',patient_groups=['a']*4,expected_training_groups=['a'],manifest_sha256='a'*64)
    assert s['scale']==[1,1,1]
    np.testing.assert_array_equal(transform_rr(raw,valid,s)[:,:3],np.zeros((4,3)))
    bad=raw.copy();bad[0,0]=np.nan
    with pytest.raises(ValueError):transform_rr(bad,valid,s)
    valid[0,0]=False
    with pytest.raises(ValueError):transform_rr(raw,valid,s)  # Missing must use zero sentinel.


def test_stable_identity_distinguishes_records_and_annotations():
    ai=np.array([2,4],dtype=np.int64);p=np.array([370,1000],dtype=np.int64)
    assert beat_ids('100',ai,p).tolist()==['100:2:370','100:4:1000']
    assert not np.array_equal(beat_ids('100',ai,p),beat_ids('101',ai,p))
    with pytest.raises(ValueError):beat_ids('100',np.array([-1]),np.array([10]))


def test_final_test_adapter_rejected_before_io():
    with pytest.raises(PermissionError):MITBIHRRDataset('missing',partition='test')


def test_configuration_freeze_and_feature_order():
    assert len(RR_FEATURE_NAMES)==6
    with pytest.raises(ValueError):RRFeatureConfig(history_intervals=9)


def test_fusion_shapes_gradients_and_same_backbone_initialization(tmp_path):
    torch.set_num_threads(2);torch.manual_seed(42);base=MITBIHBeatCNN()
    torch.manual_seed(42);model=MITBIHBeatCNNRR()
    for k,v in base.features.state_dict().items():torch.testing.assert_close(v,model.features.state_dict()[k],rtol=0,atol=0)
    assert base.parameter_count()==17797 and model.parameter_count()==17989
    x=torch.randn(4,2,252);rr=torch.zeros(4,6);rr[:,3:]=1
    logits=model(x,rr);assert logits.shape==(4,5) and torch.isfinite(logits).all()
    loss=torch.nn.CrossEntropyLoss()(logits,torch.tensor([0,1,2,3]));loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert float(model.head[0].weight.grad[:,64:].abs().sum())>0  # RR branch participates.
    path=tmp_path/'model.pt';torch.save(model.state_dict(),path)
    other=MITBIHBeatCNNRR();other.load_state_dict(torch.load(path,weights_only=True));model.eval();other.eval()
    with torch.inference_mode():torch.testing.assert_close(model(x,rr),other(x,rr),rtol=0,atol=0)
    with pytest.raises(ValueError):model(x,torch.zeros(4,5))
    with pytest.raises(TypeError):model(x.double(),rr)
    bad=rr.clone();bad[0,0]=float('inf')
    with pytest.raises(ValueError):model(x,bad)

from test_mitbih_setup import fixture_export
from src.prepare_dataset import sha256_file


@pytest.fixture
def rr_fixture(fixture_export,tmp_path):
    index,split=fixture_export;root=tmp_path/'rr';(root/'recordings').mkdir(parents=True)
    pending={};train=[];masks=[];groups=[]
    for e in index.entries:
        if e.patient_group in split['groups']['test']:continue
        part='train' if e.patient_group in split['groups']['train'] else 'validation'
        with np.load(index.root/e.file,allow_pickle=False) as z:
            a={k:z[k] for k in ('y','record_ids','patient_groups','original_symbols','annotation_positions','annotation_indices')}
        raw,valid=rr_features(a['annotation_positions'],a['annotation_positions'],360)
        a.update(rr_raw=raw,rr_valid=valid,beat_ids=beat_ids(e.record_id,a['annotation_indices'],a['annotation_positions']),
            source_rows=np.arange(e.count,dtype=np.int64),global_source_indices=e.global_start+np.arange(e.count,dtype=np.int64),
            feature_names=np.array(RR_FEATURE_NAMES),sampling_frequency=np.array(360.))
        pending[e.record_id]=(e,part,a)
        if part=='train':train.append(raw);masks.append(valid);groups.extend([e.patient_group]*e.count)
    state=fit_training_transform(np.concatenate(train),np.concatenate(masks),partition='train',patient_groups=groups,
        expected_training_groups=split['groups']['train'],manifest_sha256=index.manifest_sha256)
    (root/'training_transform.json').write_text(json.dumps(state));(root/'rr_configuration.json').write_text(json.dumps(RRFeatureConfig().to_dict()))
    rows=[]
    for rid,(e,part,a) in pending.items():
        a['RR']=transform_rr(a['rr_raw'],a['rr_valid'],state)
        path=root/f'recordings/{rid}_rr.npz';np.savez_compressed(path,**a)
        rows.append(dict(record_id=rid,patient_group=e.patient_group,count=e.count,partition=part,file=f'recordings/{rid}_rr.npz',
                         sha256=sha256_file(path),source_X_sha256=e.sha256))
    manifest=dict(status='complete',schema='mitbih-rr-sidecar-v1',RR_width=6,class_names=list('NSVFQ'),
        source_manifest_sha256=index.manifest_sha256,feature_configuration=RRFeatureConfig().to_dict(),records=rows,
        artifact_sha256={n:sha256_file(root/n) for n in ('rr_configuration.json','training_transform.json')})
    (root/'manifest.json').write_text(json.dumps(manifest))
    return index,split,root


def test_lazy_join_alignment_dtypes_masks_and_copy(rr_fixture):
    from src.mitbih_rr_dataset import collate_rr
    index,split,root=rr_fixture;ds=MITBIHRRDataset(root,index=index,split=split,max_cached_shards=1)
    assert index.read_records==[]
    sample=ds[2];expected=sample['RR'].clone()
    assert sample['X'].shape==(2,252) and sample['RR'].shape==(6,) and sample['RR'].dtype==torch.float32
    assert sample['y'].dtype==torch.long and sample['beat_id']=='100:2:820'
    assert sample['rr_valid'].dtype==torch.bool and sample['rr_raw'].dtype==torch.float64
    sample['RR'].zero_();torch.testing.assert_close(ds[2]['RR'],expected)
    batch=collate_rr([ds[0],ds[2]]);assert batch['RR'].shape==(2,6) and len(batch['beat_id'])==2
    assert index.read_records==['100']


def test_bad_sidecar_identity_or_missing_file_fails(rr_fixture):
    index,split,root=rr_fixture;ds=MITBIHRRDataset(root,index=index,split=split)
    path=root/'recordings/100_rr.npz'
    with np.load(path,allow_pickle=False) as z:a={k:z[k] for k in z.files}
    a['y']=np.roll(a['y'],1);np.savez_compressed(path,**a)
    manifest=json.loads((root/'manifest.json').read_text());manifest['records'][0]['sha256']=sha256_file(path)
    (root/'manifest.json').write_text(json.dumps(manifest))
    ds=MITBIHRRDataset(root,index=index,split=split)
    with pytest.raises(ValueError,match='misalignment'):ds[0]
    path.unlink()
    with pytest.raises(FileNotFoundError):MITBIHRRDataset(root,index=index,split=split)


def test_cannot_fit_on_changed_validation_provenance_or_tamper_transform(rr_fixture):
    index,split,root=rr_fixture
    p=root/'training_transform.json';state=json.loads(p.read_text());state['training_groups']=split['groups']['validation']
    p.write_text(json.dumps(state))
    # Valid file checksum does not make a validation-fitted transform acceptable.
    m=json.loads((root/'manifest.json').read_text());m['artifact_sha256']['training_transform.json']=sha256_file(p)
    (root/'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError,match='training groups'):MITBIHRRDataset(root,index=index,split=split)
